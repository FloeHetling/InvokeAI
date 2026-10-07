"""CLIP Tag Autocomplete — sidecar database layer.

Manages FTS5 capability probing, sidecar path derivation, writer/reader
connection setup, CTA_CASEFOLD registration, and migration execution.
"""

from __future__ import annotations

import sqlite3
from logging import Logger
from pathlib import Path

from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import (
    CtaStatus,
    CtaStatusReason,
    CtaUnavailableError,
)
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationError
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SqliteMigrator


def probe_fts5(logger: Logger) -> bool:
    """Probe whether SQLite has FTS5 support. Returns True if available."""
    try:
        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE VIRTUAL TABLE __cta_fts5_probe USING fts5(content)")
            conn.execute("DROP TABLE __cta_fts5_probe")
        return True
    except Exception as e:
        logger.warning(f"FTS5 probe failed: {e}; SQLite version: {sqlite3.sqlite_version}")
        return False


def register_cta_casefold(conn: sqlite3.Connection) -> None:
    """Register CTA_CASEFOLD function and collation on a connection."""

    def cta_casefold(value: str | None) -> str:
        if value is None:
            return ""
        return value.casefold()

    def cta_casefold_collation(left: str, right: str) -> int:
        left_cf = left.casefold()
        right_cf = right.casefold()
        return (left_cf > right_cf) - (left_cf < right_cf)

    conn.create_function("CTA_CASEFOLD", 1, cta_casefold, deterministic=True)
    conn.create_collation("CTA_CASEFOLD", cta_casefold_collation)


def validate_sidecar_file(db_path: Path, logger: Logger) -> str:
    """Validate an existing sidecar file.

    Returns:
        'fresh' — file doesn't exist or is empty.
        'compatible' — file exists with recognized CTA migration metadata.
        'incompatible' — file exists but contains unrelated SQLite data.
    """
    if not db_path.exists():
        return "fresh"

    if db_path.stat().st_size == 0:
        return "fresh"

    try:
        with sqlite3.connect(str(db_path)) as conn:
            register_cta_casefold(conn)

            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applied_migrations'")
            if cursor.fetchone() is not None:
                applied_ids = {row[0] for row in conn.execute("SELECT migration_id FROM applied_migrations")}
                if applied_ids == {"cta_0001_initial"}:
                    return "compatible"

            return "incompatible"
    except Exception as e:
        logger.warning(f"Sidecar validation failed: {e}")
        return "incompatible"


class CtaDatabase:
    """Manages the CTA sidecar database connections and migrations."""

    def __init__(self, config: InvokeAIAppConfig, logger: Logger) -> None:
        self._logger = logger
        self._config = config
        self._writer: SqliteDatabase | None = None
        self._reader: SqliteDatabase | None = None
        self._available = False
        self._status: CtaStatus = CtaStatus(available=False, reason=CtaStatusReason.DATABASE_ERROR)

    @property
    def status(self) -> CtaStatus:
        return self._status

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def writer(self) -> SqliteDatabase:
        if self._writer is None:
            raise CtaUnavailableError("CTA database not initialized")
        return self._writer

    @property
    def reader(self) -> SqliteDatabase:
        if self._reader is None:
            raise CtaUnavailableError("CTA database not initialized")
        return self._reader

    def initialize(self) -> None:
        """Initialize the CTA sidecar database. Call once at startup."""
        # Step 1: Probe FTS5
        if not probe_fts5(self._logger):
            self._status = CtaStatus(available=False, reason=CtaStatusReason.FTS5_UNAVAILABLE)
            self._logger.warning("CTA unavailable: FTS5 not supported")
            return

        # Step 2: Determine sidecar path
        # Must use db_path.parent (resolved) not db_dir (raw relative)
        use_memory = self._config.use_memory_db
        db_path = None if use_memory else self._config.db_path.parent / "cta.db"

        # Step 3: Validate existing file
        if db_path is not None:
            validation = validate_sidecar_file(db_path, self._logger)
            if validation == "incompatible":
                self._status = CtaStatus(
                    available=False,
                    reason=CtaStatusReason.DATABASE_INCOMPATIBLE,
                )
                self._logger.warning(f"CTA unavailable: incompatible sidecar at {db_path}")
                return

        # Step 4: Create writer connection
        try:
            self._writer = SqliteDatabase(db_path=db_path, logger=self._logger)
            register_cta_casefold(self._writer._conn)
        except Exception as e:
            self._status = CtaStatus(available=False, reason=CtaStatusReason.DATABASE_ERROR)
            self._logger.error(f"CTA writer connection failed: {e}")
            return

        # Step 5: Run CTA migrations on writer
        try:
            self._upgrade_legacy_migration_metadata(self._writer)
            self._run_migrations(self._writer)
        except MigrationError as e:
            self._status = CtaStatus(available=False, reason=CtaStatusReason.DATABASE_INCOMPATIBLE)
            self._logger.error(f"CTA database is incompatible: {e}")
            self._close_connections()
            return
        except Exception as e:
            self._status = CtaStatus(available=False, reason=CtaStatusReason.DATABASE_ERROR)
            self._logger.error(f"CTA migrations failed: {e}")
            self._close_connections()
            return

        # Step 6: Create reader connection
        try:
            if use_memory:
                # In-memory: share the same instance so writer and reader see the
                # same data (each sqlite3.connect(":memory:") is isolated).
                self._reader = self._writer
            else:
                self._reader = SqliteDatabase(db_path=db_path, logger=self._logger)
                register_cta_casefold(self._reader._conn)
                self._reader._conn.execute("PRAGMA query_only = ON;")
        except Exception as e:
            self._status = CtaStatus(available=False, reason=CtaStatusReason.DATABASE_ERROR)
            self._logger.error(f"CTA reader connection failed: {e}")
            self._close_connections()
            return

        # Step 7: Mark available
        self._available = True
        self._status = CtaStatus(available=True, reason=None)
        self._logger.info(f"CTA sidecar initialized at {db_path}")

    def _run_migrations(self, db: SqliteDatabase) -> None:
        """Run CTA-specific migrations."""
        from invokeai.app.services.clip_tag_autocomplete.migrations.cta_0001_initial import (
            build_migration,
        )

        migrator = SqliteMigrator(db=db)
        migrator.register_migration(build_migration())
        migrator.run_migrations()

    def _upgrade_legacy_migration_metadata(self, db: SqliteDatabase) -> None:
        """Upgrade metadata written by early CTA development builds.

        Those builds created ``applied_migrations`` with only ``migration_id``.
        Supporting that local development format avoids making an otherwise valid
        sidecar disposable while still handing future-ID validation to the shared
        migrator.
        """
        with db.transaction() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applied_migrations'")
            if cursor.fetchone() is None:
                return
            cursor.execute("PRAGMA table_info(applied_migrations)")
            columns = {row[1] for row in cursor.fetchall()}
            if "legacy_version" not in columns:
                cursor.execute("ALTER TABLE applied_migrations ADD COLUMN legacy_version INTEGER")
            if "migrated_at" not in columns:
                cursor.execute("ALTER TABLE applied_migrations ADD COLUMN migrated_at TEXT")
                cursor.execute(
                    "UPDATE applied_migrations SET migrated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW') "
                    "WHERE migrated_at IS NULL"
                )

    def _close_connections(self) -> None:
        reader = self._reader
        writer = self._writer
        self._reader = None
        self._writer = None
        if reader is not None and reader is not writer:
            reader._conn.close()
        if writer is not None:
            writer._conn.close()

    def cleanup_orphan_model_configs(self, installed_model_ids: set[str]) -> int:
        """Remove CTA model configs whose model_id is not in installed_model_ids.

        Returns the number of removed configs.
        """
        if not self._available or self._writer is None:
            return 0

        with self._writer.transaction() as cursor:
            cursor.execute("SELECT model_id FROM cta_model_configs")
            existing_ids = {row[0] for row in cursor.fetchall()}
            orphan_ids = existing_ids - installed_model_ids

            if not orphan_ids:
                return 0

            for model_id in orphan_ids:
                cursor.execute(
                    "DELETE FROM cta_model_configs WHERE model_id = ?",
                    (model_id,),
                )

            self._logger.info(f"Cleaned up {len(orphan_ids)} orphan CTA model configs")
            return len(orphan_ids)
