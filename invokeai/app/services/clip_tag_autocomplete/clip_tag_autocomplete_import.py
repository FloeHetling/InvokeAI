"""CLIP Tag Autocomplete — import session management, CSV parsing, staging, and commit."""

from __future__ import annotations

import csv
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Any, BinaryIO

from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import (
    CTA_VALID_TAG_TYPES,
    CtaImportFatalError,
    CtaImportSessionNotFoundError,
    merge_popularity,
    normalize_import_content,
)


class CtaImportState(str, Enum):
    UPLOADED = "uploaded"
    MAPPING_READY = "mapping_ready"
    PREPARED = "prepared"
    COMMITTED = "committed"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass
class CtaImportSession:
    id: str
    owner_user_id: str
    state: CtaImportState
    created_at: float
    last_access_at: float
    source_path: Path
    staging_db_path: Path | None = None
    detected_delimiter: str | None = None
    detected_header: bool | None = None
    column_count: int | None = None
    summary: CtaImportSummary | None = None


@dataclass
class CtaImportSummary:
    rows_read: int = 0
    valid_rows: int = 0
    skipped_rows: int = 0
    invalid_popularity_to_unknown: int = 0
    unknown_types_to_other: int = 0


@dataclass
class CtaImportPreview:
    session_id: str
    state: str
    preview: list[dict[str, Any]]
    summary: CtaImportSummary


@dataclass
class CtaImportResult:
    new_tags: int = 0
    existing_tags_merged: int = 0
    popularity_updated: int = 0
    type_conflicts_ignored: int = 0
    skipped_rows: int = 0


# ---------------------------------------------------------------------------
# Delimiter detection
# ---------------------------------------------------------------------------


def detect_delimiter(sample: str) -> str:
    """Detect the most likely delimiter from a sample of CSV text."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def detect_header(first_row: str, delimiter: str) -> bool:
    """Return a conservative editable initial guess for header presence."""
    rows = list(csv.reader(first_row.splitlines(), delimiter=delimiter))
    if not rows:
        return False
    normalized = {field.strip().casefold() for field in rows[0]}
    if normalized & {"tag", "name", "content", "popularity", "count", "type"}:
        return True
    if len(rows[0]) == 1:
        return False
    try:
        return csv.Sniffer().has_header(first_row)
    except csv.Error:
        return False


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


def infer_tag_type(content: str, explicit_type: str | None) -> str:
    """Resolve tag type: explicit > first-parenthesis inference > 'other'."""
    # 1. Valid explicit type wins
    if explicit_type in CTA_VALID_TAG_TYPES:
        return explicit_type

    # 2. First-parenthesis inference
    match = re.search(r"\(([^)]+)\)", content)
    if match:
        group = match.group(1)
        if group in CTA_VALID_TAG_TYPES:
            return group

    # 3. Fallback
    return "other"


# ---------------------------------------------------------------------------
# Import session manager
# ---------------------------------------------------------------------------


class CtaImportManager:
    """Manages CTA import sessions in a temporary directory."""

    SESSION_TTL = 3600  # 60 minutes

    def __init__(self, base_dir: Path, logger: Logger) -> None:
        self._base_dir = base_dir
        self._logger = logger
        self._sessions: dict[str, CtaImportSession] = {}
        self._lock = threading.RLock()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_access_at > self.SESSION_TTL]
        for sid in expired:
            self._delete_session_files(sid)
            del self._sessions[sid]

    def _delete_session_files(self, session_id: str) -> None:
        import shutil

        session_dir = self._base_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)

    def create_session(self, owner_user_id: str) -> CtaImportSession:
        with self._lock:
            self._cleanup_expired()
            session_id = str(uuid.uuid4())
            session_dir = self._base_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            now = time.time()
            session = CtaImportSession(
                id=session_id,
                owner_user_id=owner_user_id,
                state=CtaImportState.UPLOADED,
                created_at=now,
                last_access_at=now,
                source_path=session_dir / "source.csv",
            )
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str, owner_user_id: str) -> CtaImportSession:
        with self._lock:
            self._cleanup_expired()
            session = self._sessions.get(session_id)
            if not session:
                raise CtaImportSessionNotFoundError(f"Import session {session_id} not found")
            if session.owner_user_id != owner_user_id:
                raise CtaImportSessionNotFoundError("Session belongs to another user")
            session.last_access_at = time.time()
            return session

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._delete_session_files(session_id)
            self._sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Stage upload
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 8192


def stage_upload(
    session: CtaImportSession,
    upload: BinaryIO,
    logger: Logger,
) -> dict[str, Any]:
    """Spool upload to disk, inspect a bounded sample for delimiter/header/columns."""
    total_size = 0
    with open(session.source_path, "wb") as f:
        while True:
            chunk = upload.read(_CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            total_size += len(chunk)

    if total_size == 0:
        raise CtaImportFatalError("Uploaded file is empty")

    # Read bounded sample for detection (first 8KB)
    sample_bytes = min(total_size, _CHUNK_SIZE)
    with open(session.source_path, "rb") as f:
        raw_sample = f.read(sample_bytes)

    # Try UTF-8-sig decode
    try:
        sample_text = raw_sample.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CtaImportFatalError("File is not valid UTF-8") from exc

    lines = sample_text.splitlines()
    if not lines:
        raise CtaImportFatalError("File has no data rows")

    delimiter = detect_delimiter(lines[0])
    header = detect_header(sample_text, delimiter)
    parsed_sample = list(csv.reader(lines, delimiter=delimiter))
    first_data_row = parsed_sample[1] if header and len(parsed_sample) > 1 else parsed_sample[0]
    column_count = len(first_data_row)

    session.detected_delimiter = delimiter
    session.detected_header = header
    session.column_count = column_count

    # Preview: first 5-10 rows
    start = 1 if header else 0
    preview_rows = []
    preview_rows.extend(parsed_sample[start : start + 10])

    columns = parsed_sample[0] if header else [f"Column {i + 1}" for i in range(column_count)]

    session.state = CtaImportState.MAPPING_READY

    return {
        "session_id": session.id,
        "detected_delimiter": delimiter,
        "detected_header": header,
        "columns": columns,
        "sample_rows": preview_rows,
        "is_single_column": column_count == 1,
    }


# ---------------------------------------------------------------------------
# Prepare (full parse + staging DB)
# ---------------------------------------------------------------------------


def prepare_import(
    session: CtaImportSession,
    tag_column: int,
    popularity_column: int | None = None,
    type_column: int | None = None,
    delimiter: str | None = None,
    first_row_contains_column_names: bool | None = None,
    logger: Logger | None = None,
) -> CtaImportPreview:
    """Full streaming parse of the staged CSV into a temporary SQLite staging DB."""
    if session.state != CtaImportState.MAPPING_READY:
        raise CtaImportFatalError("Import session has already been prepared or is not ready")
    if not session.source_path.exists():
        raise CtaImportFatalError("Source file not found")

    delim = delimiter or session.detected_delimiter or ","
    has_header = (
        first_row_contains_column_names if first_row_contains_column_names is not None else session.detected_header
    )
    if delim not in {",", ";", "\t", "|"}:
        raise CtaImportFatalError("Unsupported delimiter")
    mapped_columns = [tag_column]
    if popularity_column is not None:
        mapped_columns.append(popularity_column)
    if type_column is not None:
        mapped_columns.append(type_column)
    if any(column < 0 for column in mapped_columns) or len(mapped_columns) != len(set(mapped_columns)):
        raise CtaImportFatalError("Import column mapping is invalid")
    if session.column_count is not None and any(column >= session.column_count for column in mapped_columns):
        raise CtaImportFatalError("A mapped import column does not exist")

    staging_path = session.source_path.parent / "staging.db"
    session.staging_db_path = staging_path

    # Create staging DB
    staging_conn = sqlite3.connect(str(staging_path))

    def staging_casefold(value: str | None) -> str:
        if value is None:
            return ""
        return value.casefold()

    def staging_casefold_collation(left: str, right: str) -> int:
        left_cf = left.casefold()
        right_cf = right.casefold()
        return (left_cf > right_cf) - (left_cf < right_cf)

    staging_conn.create_function("CTA_CASEFOLD", 1, staging_casefold, deterministic=True)
    staging_conn.create_collation("CTA_CASEFOLD", staging_casefold_collation)

    summary = CtaImportSummary()

    try:
        staging_conn.execute(
            """
            CREATE TABLE staged_tags (
                canonical_content TEXT NOT NULL,
                popularity INTEGER,
                tag_type TEXT NOT NULL,
                CHECK (popularity IS NULL OR popularity >= 0)
            )
            """
        )
        staging_conn.execute(
            """
            CREATE UNIQUE INDEX uq_staged_tags_identity
            ON staged_tags(canonical_content COLLATE CTA_CASEFOLD, tag_type)
            """
        )

        with open(
            session.source_path,
            "r",
            encoding="utf-8-sig",
            errors="strict",
            newline="",
        ) as f:
            reader = csv.reader(f, delimiter=delim, strict=True)

            for i, row in enumerate(reader):
                # Skip header row
                if i == 0 and has_header:
                    continue
                summary.rows_read += 1

                # Ensure row has enough columns
                if len(row) <= tag_column:
                    summary.skipped_rows += 1
                    continue

                raw_tag = row[tag_column].strip()
                if not raw_tag:
                    summary.skipped_rows += 1
                    continue

                # Normalize content
                canonical = normalize_import_content(raw_tag)
                if not canonical:
                    summary.skipped_rows += 1
                    continue

                # Parse popularity
                pop: int | None = None
                if popularity_column is not None and len(row) > popularity_column:
                    raw_pop = row[popularity_column].strip()
                    if raw_pop:
                        try:
                            pop = int(raw_pop)
                            if pop < 0:
                                pop = None
                                summary.invalid_popularity_to_unknown += 1
                        except ValueError:
                            pop = None
                            summary.invalid_popularity_to_unknown += 1

                # Resolve tag type
                explicit_type: str | None = None
                if type_column is not None and len(row) > type_column:
                    raw_type = row[type_column].strip()
                    if raw_type:
                        explicit_type = raw_type

                tag_type = infer_tag_type(canonical, explicit_type)
                if explicit_type and explicit_type not in CTA_VALID_TAG_TYPES and tag_type == "other":
                    summary.unknown_types_to_other += 1

                # Insert/upsert into staging
                staging_conn.execute(
                    """INSERT INTO staged_tags (canonical_content, popularity, tag_type)
                       VALUES (?, ?, ?)
                       ON CONFLICT(canonical_content COLLATE CTA_CASEFOLD, tag_type)
                       DO UPDATE SET popularity = CASE
                           WHEN excluded.popularity IS NULL THEN staged_tags.popularity
                           WHEN staged_tags.popularity IS NULL THEN excluded.popularity
                           ELSE MAX(staged_tags.popularity, excluded.popularity)
                       END""",
                    (canonical, pop, tag_type),
                )
        staging_conn.commit()
        summary.valid_rows = staging_conn.execute("SELECT COUNT(*) FROM staged_tags").fetchone()[0]
        preview_rows = [
            {
                "canonical_content": row[0],
                "popularity": row[1],
                "tag_type": row[2],
            }
            for row in staging_conn.execute(
                "SELECT canonical_content, popularity, tag_type "
                "FROM staged_tags ORDER BY canonical_content COLLATE CTA_CASEFOLD, tag_type LIMIT 20"
            ).fetchall()
        ]
    except UnicodeDecodeError as exc:
        staging_conn.rollback()
        raise CtaImportFatalError("File contains invalid UTF-8 characters") from exc
    except csv.Error as exc:
        staging_conn.rollback()
        raise CtaImportFatalError(f"CSV parsing error: {exc}") from exc
    except sqlite3.Error as exc:
        staging_conn.rollback()
        raise CtaImportFatalError(f"Unable to stage imported tags: {exc}") from exc
    finally:
        staging_conn.close()

    session.state = CtaImportState.PREPARED
    session.summary = summary

    return CtaImportPreview(
        session_id=session.id,
        state="prepared",
        preview=preview_rows,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Commit (import from staging into real CTA DB)
# ---------------------------------------------------------------------------


def commit_import(
    session: CtaImportSession,
    writer_conn: sqlite3.Connection,
    destination_type: str,  # "uncategorized", "new_set", "existing_set"
    destination_name: str | None = None,
    destination_tag_set_id: str | None = None,
    import_mode: str = "merge",  # "merge" or "replace"
) -> CtaImportResult:
    """Commit staged tags into the real CTA database."""
    if session.state != CtaImportState.PREPARED:
        raise CtaImportFatalError("Session is not in PREPARED state")

    if not session.staging_db_path or not session.staging_db_path.exists():
        raise CtaImportFatalError("Staging database not found")
    if destination_type not in {"uncategorized", "new_set", "existing_set"}:
        raise CtaImportFatalError("Invalid import destination")
    if destination_type == "new_set" and not destination_name:
        raise CtaImportFatalError("A new tag set name is required")
    if destination_type == "existing_set" and not destination_tag_set_id:
        raise CtaImportFatalError("An existing tag set is required")
    if import_mode not in {"merge", "replace"}:
        raise CtaImportFatalError("Invalid import mode")

    if destination_type == "existing_set":
        existing_set = writer_conn.execute(
            "SELECT id FROM cta_tag_sets WHERE id = ?", (destination_tag_set_id,)
        ).fetchone()
        if existing_set is None:
            raise CtaImportFatalError("Destination tag set not found")

    staging_conn = sqlite3.connect(str(session.staging_db_path))

    result = CtaImportResult()

    try:
        # Create resolved-ID temp table
        writer_conn.execute("CREATE TEMP TABLE IF NOT EXISTS cta_import_resolved_tag_ids (tag_id TEXT PRIMARY KEY)")
        writer_conn.execute("DELETE FROM cta_import_resolved_tag_ids")

        # Stream staged rows
        cursor = staging_conn.execute("SELECT canonical_content, popularity, tag_type FROM staged_tags")

        for row in cursor:
            canonical, pop, tag_type = row[0], row[1], row[2]
            resolved_id = _resolve_imported_tag(writer_conn, canonical, tag_type, pop, result)
            if isinstance(resolved_id, list):
                for rid in resolved_id:
                    writer_conn.execute(
                        "INSERT OR IGNORE INTO cta_import_resolved_tag_ids (tag_id) VALUES (?)",
                        (rid,),
                    )
            else:
                writer_conn.execute(
                    "INSERT OR IGNORE INTO cta_import_resolved_tag_ids (tag_id) VALUES (?)",
                    (resolved_id,),
                )

        # Create destination tag set if needed
        if destination_type == "new_set" and destination_name:
            tag_set_id = str(uuid.uuid4())
            writer_conn.execute(
                "INSERT INTO cta_tag_sets (id, name) VALUES (?, ?)",
                (tag_set_id, destination_name),
            )
            destination_tag_set_id = tag_set_id

        # Apply membership
        if destination_type == "uncategorized":
            pass  # No membership rows
        elif destination_type in ("new_set", "existing_set") and destination_tag_set_id:
            if import_mode == "replace":
                writer_conn.execute(
                    """DELETE FROM cta_tag_set_tags
                       WHERE tag_set_id = ?
                       AND tag_id NOT IN (SELECT tag_id FROM cta_import_resolved_tag_ids)""",
                    (destination_tag_set_id,),
                )

            writer_conn.execute(
                """INSERT OR IGNORE INTO cta_tag_set_tags (tag_set_id, tag_id)
                   SELECT ?, tag_id FROM cta_import_resolved_tag_ids""",
                (destination_tag_set_id,),
            )

        # Drop temp table
        writer_conn.execute("DROP TABLE IF EXISTS cta_import_resolved_tag_ids")

        result.skipped_rows = session.summary.skipped_rows if session.summary is not None else 0

    finally:
        staging_conn.close()

    return result


def _resolve_imported_tag(
    conn: sqlite3.Connection,
    canonical_content: str,
    incoming_type: str,
    incoming_popularity: int | None,
    result: CtaImportResult,
) -> str | list[str]:
    """Resolve an imported tag against existing tags. Returns resolved tag ID(s)."""
    # Find existing same-content tags
    cur = conn.execute(
        "SELECT id, canonical_content, tag_type, popularity FROM cta_tags "
        "WHERE canonical_content COLLATE CTA_CASEFOLD = ? "
        "ORDER BY tag_type, id",
        (canonical_content,),
    )
    existing_rows = cur.fetchall()

    if not existing_rows:
        # New tag
        tag_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, ?, ?, ?)",
            (tag_id, canonical_content, incoming_popularity, incoming_type),
        )
        result.new_tags += 1
        return tag_id

    # Classify existing tags and exact semantic identity.
    known_tags = [r for r in existing_rows if r[2] != "other"]
    other_tags = [r for r in existing_rows if r[2] == "other"]
    exact = next((r for r in existing_rows if r[2] == incoming_type), None)

    if exact is not None:
        target_id = str(exact[0])
        existing_popularity = exact[3]
        merged_popularity = merge_popularity(existing_popularity, incoming_popularity)

        # A known non-artist identity absorbs the fallback `other` identity while
        # preserving its memberships. Artist intentionally remains ambiguous.
        if incoming_type not in {"other", "artist"} and other_tags:
            other = other_tags[0]
            merged_popularity = merge_popularity(merged_popularity, other[3])
            conn.execute(
                "INSERT OR IGNORE INTO cta_tag_set_tags (tag_set_id, tag_id) "
                "SELECT tag_set_id, ? FROM cta_tag_set_tags WHERE tag_id = ?",
                (target_id, other[0]),
            )
            conn.execute("DELETE FROM cta_tags WHERE id = ?", (other[0],))

        if merged_popularity != existing_popularity:
            conn.execute(
                "UPDATE cta_tags SET popularity = ? WHERE id = ?",
                (merged_popularity, target_id),
            )
            result.popularity_updated += 1
        result.existing_tags_merged += 1
        return target_id

    # Incoming `other` never creates a fallback identity when known meanings exist.
    if incoming_type == "other" and known_tags:
        resolved_ids: list[str] = []
        for target in known_tags:
            resolved_ids.append(str(target[0]))
            new_pop = merge_popularity(target[3], incoming_popularity)
            if new_pop != target[3]:
                conn.execute(
                    "UPDATE cta_tags SET popularity = ? WHERE id = ?",
                    (new_pop, target[0]),
                )
                result.popularity_updated += 1
        result.existing_tags_merged += 1
        if len(resolved_ids) > 1:
            result.type_conflicts_ignored += 1
        return resolved_ids

    # Known non-artist identities upgrade `other` in place. Artist names preserve
    # the fallback identity because they are intentionally ambiguous.
    if incoming_type not in {"other", "artist"} and other_tags:
        other = other_tags[0]
        new_pop = merge_popularity(other[3], incoming_popularity)
        conn.execute(
            "UPDATE cta_tags SET tag_type = ?, popularity = ? WHERE id = ?",
            (incoming_type, new_pop, other[0]),
        )
        if new_pop != other[3]:
            result.popularity_updated += 1
        result.existing_tags_merged += 1
        return str(other[0])

    # Artist and character are the one intentional known-type coexistence pair.
    if incoming_type in {"artist", "character"} and any(known[2] in {"artist", "character"} for known in known_tags):
        tag_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, ?, ?, ?)",
            (tag_id, canonical_content, incoming_popularity, incoming_type),
        )
        result.new_tags += 1
        return tag_id

    # Other conflicting known types do not create or overwrite a classification.
    if known_tags:
        existing = known_tags[0]
        new_pop = merge_popularity(existing[3], incoming_popularity)
        if new_pop != existing[3]:
            conn.execute(
                "UPDATE cta_tags SET popularity = ? WHERE id = ?",
                (new_pop, existing[0]),
            )
            result.popularity_updated += 1
        result.type_conflicts_ignored += 1
        result.existing_tags_merged += 1
        return str(existing[0])

    # Fallback: create as other
    tag_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, ?, ?, ?)",
        (tag_id, canonical_content, incoming_popularity, incoming_type),
    )
    result.new_tags += 1
    return tag_id
