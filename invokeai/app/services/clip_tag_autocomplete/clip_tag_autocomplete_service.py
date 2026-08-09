from __future__ import annotations

import sqlite3
import tempfile
import threading
from logging import Logger
from pathlib import Path
from typing import BinaryIO

from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import (
    CTA_VALID_TAG_TYPES,
    CtaAutocompleteCandidate,
    CtaAutocompleteFilter,
    CtaBulkResult,
    CtaBulkTagRequest,
    CtaConflictError,
    CtaImportPreviewDTO,
    CtaImportResultDTO,
    CtaImportStageDTO,
    CtaModelConfigDTO,
    CtaNotFoundError,
    CtaStatus,
    CtaSyntaxProfileDTO,
    CtaTagDetailDTO,
    CtaTagDTO,
    CtaTagMutationResult,
    CtaTagPage,
    CtaTagSetDTO,
    CtaUnavailableError,
    CtaWriteBusyError,
    merge_popularity,
)
from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_database import CtaDatabase


class ClipTagAutocompleteService:
    """High-level CTA service orchestrating autocomplete, CRUD, import, and model config."""

    def __init__(self, cta_db: CtaDatabase, logger: Logger) -> None:
        self._db = cta_db
        self._logger = logger
        self._write_lock = threading.Lock()
        self._pending_cleanup_lock = threading.Lock()
        self._pending_model_cleanup: set[str] = set()

        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_import import CtaImportManager

        self._import_temp_dir = tempfile.TemporaryDirectory(prefix="invokeai-cta-imports-")
        self._import_manager = CtaImportManager(
            base_dir=Path(self._import_temp_dir.name),
            logger=logger,
        )

    @property
    def status(self) -> CtaStatus:
        return self._db.status

    @property
    def is_available(self) -> bool:
        return self._db.is_available

    def _assert_available(self) -> None:
        if not self.is_available:
            raise CtaUnavailableError(f"CTA unavailable: {self._db.status.reason}")

    def _acquire_write(self) -> None:
        if not self._write_lock.acquire(blocking=False):
            raise CtaWriteBusyError("Another CTA data operation is already in progress.")

    def _release_write(self) -> None:
        try:
            with self._pending_cleanup_lock:
                pending = set(self._pending_model_cleanup)
            if pending and self.is_available:
                try:
                    with self._db.writer.transaction() as cursor:
                        cursor.executemany(
                            "DELETE FROM cta_model_configs WHERE model_id = ?",
                            [(model_id,) for model_id in pending],
                        )
                    with self._pending_cleanup_lock:
                        self._pending_model_cleanup.difference_update(pending)
                except Exception as e:
                    self._logger.warning(f"CTA deferred model config cleanup failed: {e}")
        finally:
            self._write_lock.release()

    # ------------------------------------------------------------------
    # Autocomplete
    # ------------------------------------------------------------------

    def autocomplete(
        self,
        query: str,
        model_id: str | None,
        tag_filter: CtaAutocompleteFilter | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[CtaAutocompleteCandidate]:
        """Search for tags matching the query.

        Unfiltered requests preserve the lightweight 20-result autocomplete.
        Filtered requests return 50-result pages and may be continued with offset.
        """
        self._assert_available()

        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import (
            render_cta_tag,
        )
        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_search import (
            build_fts_phrase,
            normalize_autocomplete_query,
        )

        normalized = normalize_autocomplete_query(query)
        if len(normalized) < 2 or not any(character.isalnum() for character in normalized):
            return []

        fts_query = build_fts_phrase(normalized)
        page_limit = 20 if tag_filter is None else max(1, min(limit, 100))
        page_offset = 0 if tag_filter is None else max(0, offset)

        profile: CtaSyntaxProfileDTO | None = None
        if model_id:
            with self._db.reader.transaction() as cursor:
                cursor.execute(
                    """SELECT sp.id, sp.name, sp.spaces_to_underscores, sp.escape_parentheses,
                              sp.escape_colons, sp.append_type_parentheses, sp.prefix_artist_with_by
                       FROM cta_model_configs mc
                       JOIN cta_syntax_profiles sp ON sp.id = mc.syntax_profile_id
                       WHERE mc.model_id = ?""",
                    (model_id,),
                )
                row = cursor.fetchone()
                if row:
                    profile = CtaSyntaxProfileDTO(
                        id=row[0],
                        name=row[1],
                        spaces_to_underscores=bool(row[2]),
                        escape_parentheses=bool(row[3]),
                        escape_colons=bool(row[4]),
                        append_type_parentheses=bool(row[5]),
                        prefix_artist_with_by=bool(row[6]),
                    )

        with self._db.reader.transaction() as cursor:
            cursor.execute(
                """
                WITH matches AS (
                    SELECT tag_id
                    FROM cta_tags_fts
                    WHERE cta_tags_fts MATCH :fts_query
                ),
                ranked AS (
                    SELECT
                        t.id,
                        t.canonical_content,
                        t.popularity,
                        t.tag_type,
                        CASE
                            WHEN CTA_CASEFOLD(t.canonical_content) = :normalized THEN 0
                            WHEN substr(CTA_CASEFOLD(t.canonical_content), 1, length(:normalized)) = :normalized THEN 1
                            ELSE 2
                        END AS text_rank,
                        CASE
                            WHEN :model_id IS NOT NULL AND EXISTS (
                                SELECT 1
                                FROM cta_model_tag_sets mts
                                JOIN cta_tag_set_tags tst
                                    ON tst.tag_set_id = mts.tag_set_id
                                WHERE mts.model_id = :model_id
                                  AND tst.tag_id = t.id
                                LIMIT 1
                            ) THEN 0
                            ELSE 1
                        END AS model_rank
                    FROM matches m
                    JOIN cta_tags t ON t.id = m.tag_id
                    WHERE
                        :tag_filter IS NULL
                        OR (:tag_filter = 'artist' AND t.tag_type IN ('artist', 'copyright'))
                        OR (:tag_filter = 'character' AND t.tag_type = 'character')
                        OR (
                            :tag_filter = 'other'
                            AND t.tag_type NOT IN ('artist', 'copyright', 'character')
                        )
                )
                SELECT
                    id,
                    canonical_content,
                    popularity,
                    tag_type
                FROM ranked
                ORDER BY
                    text_rank ASC,
                    model_rank ASC,
                    CASE WHEN popularity IS NULL THEN 1 ELSE 0 END ASC,
                    popularity DESC,
                    canonical_content COLLATE CTA_CASEFOLD ASC,
                    tag_type ASC,
                    id ASC
                LIMIT :limit
                OFFSET :offset
                """,
                {
                    "fts_query": fts_query,
                    "normalized": normalized,
                    "model_id": model_id,
                    "tag_filter": tag_filter,
                    "limit": page_limit,
                    "offset": page_offset,
                },
            )

            results = []
            for row in cursor.fetchall():
                tag_id, canonical, pop, tag_type = row[0], row[1], row[2], row[3]
                rendered = render_cta_tag(canonical, tag_type, profile)
                results.append(
                    CtaAutocompleteCandidate(
                        id=tag_id,
                        canonical_content=canonical,
                        popularity=pop,
                        tag_type=tag_type,
                        rendered_content=rendered,
                    )
                )

            return results

    # ------------------------------------------------------------------
    # Tags CRUD
    # ------------------------------------------------------------------

    def list_tags(
        self,
        q: str | None = None,
        tag_type: str | None = None,
        tag_set_id: str | None = None,
        uncategorized: bool | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> CtaTagPage:
        """List tags with server-side filtering and cursor pagination."""
        self._assert_available()

        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_search import (
            build_keyset_condition,
            decode_cursor,
            encode_cursor,
        )

        if tag_set_id and uncategorized:
            raise ValueError("tag_set_id and uncategorized are mutually exclusive")

        limit = max(1, min(limit, 100))

        with self._db.reader.transaction() as txn:
            base_where_clause, base_params = self._build_tag_filter(q, tag_type, tag_set_id, bool(uncategorized))
            data_conditions = [base_where_clause]
            data_params = dict(base_params)

            if cursor:
                cursor_data = decode_cursor(cursor)
                keyset_cond, keyset_params = build_keyset_condition(cursor_data, "t")
                data_conditions.append(f"({keyset_cond})")
                data_params.update(keyset_params)

            count_sql = f"SELECT COUNT(*) FROM cta_tags t WHERE {base_where_clause}"
            txn.execute(count_sql, base_params)
            total_count = txn.fetchone()[0]

            data_where_clause = " AND ".join(data_conditions)
            data_sql = f"""
                SELECT t.id, t.canonical_content, t.popularity, t.tag_type
                FROM cta_tags t
                WHERE {data_where_clause}
                ORDER BY
                    t.canonical_content COLLATE CTA_CASEFOLD ASC,
                    t.tag_type ASC,
                    t.id ASC
                LIMIT :limit_plus_one
            """
            data_params["limit_plus_one"] = limit + 1
            txn.execute(data_sql, data_params)

            rows = txn.fetchall()
            has_next = len(rows) > limit
            items = rows[:limit]

            next_cursor = None
            if has_next and items:
                last = items[-1]
                next_cursor = encode_cursor(
                    {
                        "canonical_content": last[1],
                        "tag_type": last[3],
                        "id": last[0],
                    }
                )

            return CtaTagPage(
                items=[
                    CtaTagDTO(
                        id=row[0],
                        canonical_content=row[1],
                        popularity=row[2],
                        tag_type=row[3],
                    )
                    for row in items
                ],
                next_cursor=next_cursor,
                total_count=total_count,
            )

    def get_tag(self, tag_id: str) -> CtaTagDetailDTO:
        """Get a single tag with its tag-set memberships."""
        self._assert_available()

        with self._db.reader.transaction() as txn:
            txn.execute(
                "SELECT id, canonical_content, popularity, tag_type FROM cta_tags WHERE id = ?",
                (tag_id,),
            )
            row = txn.fetchone()
            if not row:
                raise CtaNotFoundError(f"Tag {tag_id} not found")

            txn.execute(
                "SELECT tag_set_id FROM cta_tag_set_tags WHERE tag_id = ?",
                (tag_id,),
            )
            tag_set_ids = [r[0] for r in txn.fetchall()]

            return CtaTagDetailDTO(
                id=row[0],
                canonical_content=row[1],
                popularity=row[2],
                tag_type=row[3],
                tag_set_ids=tag_set_ids,
            )

    def update_tag(
        self,
        tag_id: str,
        canonical_content: str | None = None,
        tag_type: str | None = None,
        popularity: int | None = None,
        popularity_supplied: bool = False,
        tag_set_ids: list[str] | None = None,
        tag_set_ids_supplied: bool = False,
    ) -> CtaTagMutationResult:
        """Update a tag. Auto-merges on semantic identity collision."""
        self._assert_available()
        self._acquire_write()
        try:
            with self._db.writer.transaction() as txn:
                txn.execute(
                    "SELECT id, canonical_content, popularity, tag_type FROM cta_tags WHERE id = ?",
                    (tag_id,),
                )
                row = txn.fetchone()
                if not row:
                    raise CtaNotFoundError(f"Tag {tag_id} not found")

                current_content, current_type = row[1], row[3]
                if canonical_content is not None:
                    from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import (
                        normalize_import_content,
                    )

                    canonical_content = normalize_import_content(canonical_content)
                    if not canonical_content:
                        raise ValueError("Canonical content must not be empty")
                new_content = canonical_content if canonical_content is not None else current_content
                new_type = tag_type if tag_type is not None else current_type
                if new_type not in CTA_VALID_TAG_TYPES:
                    raise ValueError(f"Invalid tag type: {new_type}")

                normalized_tag_set_ids = list(dict.fromkeys(tag_set_ids or []))
                if tag_set_ids_supplied:
                    for tag_set_id in normalized_tag_set_ids:
                        txn.execute("SELECT id FROM cta_tag_sets WHERE id = ?", (tag_set_id,))
                        if txn.fetchone() is None:
                            raise CtaNotFoundError(f"Tag set {tag_set_id} not found")

                if popularity_supplied:
                    txn.execute("UPDATE cta_tags SET popularity = ? WHERE id = ?", (popularity, tag_id))

                txn.execute(
                    "SELECT id FROM cta_tags WHERE canonical_content COLLATE CTA_CASEFOLD = ? AND tag_type = ? AND id != ?",
                    (new_content.casefold(), new_type, tag_id),
                )
                collision = txn.fetchone()

                if collision:
                    target_id = collision[0]
                    self._merge_tag_into_target(txn, tag_id, target_id)
                    merged = True
                    result_id = target_id
                else:
                    txn.execute(
                        "UPDATE cta_tags SET canonical_content = ?, tag_type = ? WHERE id = ?",
                        (new_content, new_type, tag_id),
                    )
                    merged = False
                    result_id = tag_id

                if tag_set_ids_supplied:
                    txn.execute("DELETE FROM cta_tag_set_tags WHERE tag_id = ?", (result_id,))
                    for ts_id in normalized_tag_set_ids:
                        txn.execute(
                            "INSERT OR IGNORE INTO cta_tag_set_tags (tag_set_id, tag_id) VALUES (?, ?)",
                            (ts_id, result_id),
                        )

                txn.execute(
                    "SELECT id, canonical_content, popularity, tag_type FROM cta_tags WHERE id = ?",
                    (result_id,),
                )
                final = txn.fetchone()
                txn.execute(
                    "SELECT tag_set_id FROM cta_tag_set_tags WHERE tag_id = ?",
                    (result_id,),
                )
                final_sets = [r[0] for r in txn.fetchall()]

                return CtaTagMutationResult(
                    tag=CtaTagDetailDTO(
                        id=final[0],
                        canonical_content=final[1],
                        popularity=final[2],
                        tag_type=final[3],
                        tag_set_ids=final_sets,
                    ),
                    merged=merged,
                )
        finally:
            self._release_write()

    def delete_tag(self, tag_id: str) -> None:
        """Delete a tag and cascade its tag-set memberships."""
        self._assert_available()
        self._acquire_write()
        try:
            with self._db.writer.transaction() as txn:
                txn.execute("DELETE FROM cta_tags WHERE id = ?", (tag_id,))
                if txn.rowcount == 0:
                    raise CtaNotFoundError(f"Tag {tag_id} not found")
        finally:
            self._release_write()

    def _merge_tag_into_target(
        self,
        txn: sqlite3.Cursor,
        source_id: str,
        target_id: str,
    ) -> None:
        """Merge source tag into target: max popularity, union memberships, delete source."""
        txn.execute("SELECT popularity FROM cta_tags WHERE id = ?", (source_id,))
        source_pop = txn.fetchone()[0]
        txn.execute("SELECT popularity FROM cta_tags WHERE id = ?", (target_id,))
        target_pop = txn.fetchone()[0]

        merged_pop = merge_popularity(target_pop, source_pop)
        txn.execute("UPDATE cta_tags SET popularity = ? WHERE id = ?", (merged_pop, target_id))

        txn.execute(
            """INSERT OR IGNORE INTO cta_tag_set_tags (tag_set_id, tag_id)
               SELECT tag_set_id, :target_id FROM cta_tag_set_tags WHERE tag_id = :source_id""",
            {"target_id": target_id, "source_id": source_id},
        )

        txn.execute("DELETE FROM cta_tags WHERE id = ?", (source_id,))

    def _cleanup_empty_model_configs(self, txn: sqlite3.Cursor) -> None:
        txn.execute(
            """DELETE FROM cta_model_configs
               WHERE syntax_profile_id IS NULL
               AND model_id NOT IN (
                   SELECT DISTINCT model_id FROM cta_model_tag_sets
               )"""
        )

    def bulk_mutate_tags(self, request: CtaBulkTagRequest) -> CtaBulkResult:
        self._assert_available()
        self._acquire_write()
        try:
            with self._db.writer.transaction() as txn:
                txn.execute("CREATE TEMP TABLE IF NOT EXISTS cta_selected_tag_ids (tag_id TEXT PRIMARY KEY)")
                txn.execute("DELETE FROM cta_selected_tag_ids")

                if request.selection.mode == "ids":
                    txn.executemany(
                        "INSERT OR IGNORE INTO cta_selected_tag_ids (tag_id) SELECT id FROM cta_tags WHERE id = ?",
                        [(tag_id,) for tag_id in request.selection.ids],
                    )
                else:
                    condition, params = self._build_tag_filter(
                        q=request.selection.filter.q,
                        tag_type=request.selection.filter.tag_type,
                        tag_set_id=request.selection.filter.tag_set_id,
                        uncategorized=request.selection.filter.uncategorized,
                    )
                    txn.execute(
                        "INSERT OR IGNORE INTO cta_selected_tag_ids (tag_id) "
                        f"SELECT t.id FROM cta_tags t WHERE {condition}",
                        params,
                    )

                txn.execute("SELECT COUNT(*) FROM cta_selected_tag_ids")
                selected_count = int(txn.fetchone()[0])
                affected_count = 0
                merged_count = 0
                operation = request.operation

                if operation.type == "add_to_set":
                    txn.execute("SELECT id FROM cta_tag_sets WHERE id = ?", (operation.tag_set_id,))
                    if txn.fetchone() is None:
                        raise CtaNotFoundError(f"Tag set {operation.tag_set_id} not found")
                    txn.execute(
                        "INSERT OR IGNORE INTO cta_tag_set_tags (tag_set_id, tag_id) "
                        "SELECT ?, tag_id FROM cta_selected_tag_ids",
                        (operation.tag_set_id,),
                    )
                    affected_count = max(txn.rowcount, 0)
                elif operation.type == "remove_from_set":
                    txn.execute("SELECT id FROM cta_tag_sets WHERE id = ?", (operation.tag_set_id,))
                    if txn.fetchone() is None:
                        raise CtaNotFoundError(f"Tag set {operation.tag_set_id} not found")
                    txn.execute(
                        "DELETE FROM cta_tag_set_tags WHERE tag_set_id = ? "
                        "AND tag_id IN (SELECT tag_id FROM cta_selected_tag_ids)",
                        (operation.tag_set_id,),
                    )
                    affected_count = max(txn.rowcount, 0)
                elif operation.type == "delete":
                    txn.execute("DELETE FROM cta_tags WHERE id IN (SELECT tag_id FROM cta_selected_tag_ids)")
                    affected_count = max(txn.rowcount, 0)
                elif operation.type == "set_type":
                    selected_cursor = txn.connection.execute(
                        "SELECT t.id, t.canonical_content, t.tag_type "
                        "FROM cta_tags t JOIN cta_selected_tag_ids s ON s.tag_id = t.id "
                        "ORDER BY t.id"
                    )
                    while rows := selected_cursor.fetchmany(500):
                        for row in rows:
                            source_id, canonical_content, current_type = row
                            if current_type == operation.tag_type:
                                continue
                            target = txn.connection.execute(
                                "SELECT id FROM cta_tags "
                                "WHERE canonical_content COLLATE CTA_CASEFOLD = ? AND tag_type = ? AND id != ?",
                                (canonical_content, operation.tag_type, source_id),
                            ).fetchone()
                            if target is not None:
                                self._merge_tag_into_target(txn, source_id, target[0])
                                merged_count += 1
                            else:
                                txn.execute(
                                    "UPDATE cta_tags SET tag_type = ? WHERE id = ?",
                                    (operation.tag_type, source_id),
                                )
                            affected_count += 1

                txn.execute("DELETE FROM cta_selected_tag_ids")
                return CtaBulkResult(
                    selected_count=selected_count,
                    affected_count=affected_count,
                    merged_count=merged_count,
                )
        finally:
            self._release_write()

    def _build_tag_filter(
        self,
        q: str | None,
        tag_type: str | None,
        tag_set_id: str | None,
        uncategorized: bool,
    ) -> tuple[str, dict[str, object]]:
        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_search import (
            build_fts_phrase,
            normalize_autocomplete_query,
        )

        conditions: list[str] = []
        params: dict[str, object] = {}
        if q:
            normalized = normalize_autocomplete_query(q)
            if len(normalized) >= 2 and any(character.isalnum() for character in normalized):
                conditions.append("t.id IN (SELECT tag_id FROM cta_tags_fts WHERE cta_tags_fts MATCH :fts_query)")
                params["fts_query"] = build_fts_phrase(normalized)
        if tag_type:
            if tag_type not in CTA_VALID_TAG_TYPES:
                raise ValueError(f"Invalid tag type: {tag_type}")
            conditions.append("t.tag_type = :tag_type")
            params["tag_type"] = tag_type
        if tag_set_id and uncategorized:
            raise ValueError("tag_set_id and uncategorized are mutually exclusive")
        if tag_set_id:
            conditions.append(
                "EXISTS (SELECT 1 FROM cta_tag_set_tags tst WHERE tst.tag_id = t.id AND tst.tag_set_id = :tag_set_id)"
            )
            params["tag_set_id"] = tag_set_id
        if uncategorized:
            conditions.append("NOT EXISTS (SELECT 1 FROM cta_tag_set_tags tst WHERE tst.tag_id = t.id)")
        return (" AND ".join(conditions) if conditions else "1=1"), params

    # ------------------------------------------------------------------
    # Tag Sets CRUD
    # ------------------------------------------------------------------

    def list_tag_sets(self) -> list[CtaTagSetDTO]:
        self._assert_available()
        with self._db.reader.transaction() as txn:
            txn.execute(
                """
                SELECT ts.id, ts.name,
                       COUNT(DISTINCT tst.tag_id) as tag_count,
                       COUNT(DISTINCT mts.model_id) as model_count
                FROM cta_tag_sets ts
                LEFT JOIN cta_tag_set_tags tst ON tst.tag_set_id = ts.id
                LEFT JOIN cta_model_tag_sets mts ON mts.tag_set_id = ts.id
                GROUP BY ts.id, ts.name
                ORDER BY ts.name COLLATE CTA_CASEFOLD ASC
                """
            )
            return [CtaTagSetDTO(id=r[0], name=r[1], tag_count=r[2], model_count=r[3]) for r in txn.fetchall()]

    def create_tag_set(self, name: str) -> CtaTagSetDTO:
        self._assert_available()
        name = name.strip()
        if not name:
            raise ValueError("Tag set name must not be empty")
        self._acquire_write()
        try:
            import uuid

            with self._db.writer.transaction() as txn:
                tag_set_id = str(uuid.uuid4())
                try:
                    txn.execute(
                        "INSERT INTO cta_tag_sets (id, name) VALUES (?, ?)",
                        (tag_set_id, name),
                    )
                except sqlite3.IntegrityError as e:
                    raise CtaConflictError(f"Tag set '{name}' already exists") from e
                return CtaTagSetDTO(id=tag_set_id, name=name)
        finally:
            self._release_write()

    def update_tag_set(self, tag_set_id: str, name: str) -> CtaTagSetDTO:
        self._assert_available()
        name = name.strip()
        if not name:
            raise ValueError("Tag set name must not be empty")
        self._acquire_write()
        try:
            with self._db.writer.transaction() as txn:
                txn.execute("SELECT id, name FROM cta_tag_sets WHERE id = ?", (tag_set_id,))
                row = txn.fetchone()
                if not row:
                    raise CtaNotFoundError(f"Tag set {tag_set_id} not found")
                try:
                    txn.execute(
                        "UPDATE cta_tag_sets SET name = ? WHERE id = ?",
                        (name, tag_set_id),
                    )
                except sqlite3.IntegrityError as e:
                    raise CtaConflictError(f"Tag set '{name}' already exists") from e
                return CtaTagSetDTO(id=tag_set_id, name=name)
        finally:
            self._release_write()

    def delete_tag_set(self, tag_set_id: str) -> None:
        self._assert_available()
        self._acquire_write()
        try:
            with self._db.writer.transaction() as txn:
                txn.execute("DELETE FROM cta_tag_sets WHERE id = ?", (tag_set_id,))
                if txn.rowcount == 0:
                    raise CtaNotFoundError(f"Tag set {tag_set_id} not found")
                self._cleanup_empty_model_configs(txn)
        finally:
            self._release_write()

    # ------------------------------------------------------------------
    # Syntax Profiles CRUD
    # ------------------------------------------------------------------

    def list_syntax_profiles(self) -> list[CtaSyntaxProfileDTO]:
        self._assert_available()
        with self._db.reader.transaction() as txn:
            txn.execute(
                """
                SELECT id, name, spaces_to_underscores, escape_parentheses,
                       escape_colons, append_type_parentheses, prefix_artist_with_by
                FROM cta_syntax_profiles
                ORDER BY name COLLATE CTA_CASEFOLD ASC
                """
            )
            return [
                CtaSyntaxProfileDTO(
                    id=r[0],
                    name=r[1],
                    spaces_to_underscores=bool(r[2]),
                    escape_parentheses=bool(r[3]),
                    escape_colons=bool(r[4]),
                    append_type_parentheses=bool(r[5]),
                    prefix_artist_with_by=bool(r[6]),
                )
                for r in txn.fetchall()
            ]

    def create_syntax_profile(
        self,
        name: str,
        spaces_to_underscores: bool = True,
        escape_parentheses: bool = True,
        escape_colons: bool = True,
        append_type_parentheses: bool = False,
        prefix_artist_with_by: bool = False,
    ) -> CtaSyntaxProfileDTO:
        self._assert_available()
        name = name.strip()
        if not name:
            raise ValueError("Syntax profile name must not be empty")
        self._acquire_write()
        try:
            import uuid

            with self._db.writer.transaction() as txn:
                profile_id = str(uuid.uuid4())
                try:
                    txn.execute(
                        """INSERT INTO cta_syntax_profiles
                           (id, name, spaces_to_underscores, escape_parentheses,
                            escape_colons, append_type_parentheses, prefix_artist_with_by)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            profile_id,
                            name,
                            int(spaces_to_underscores),
                            int(escape_parentheses),
                            int(escape_colons),
                            int(append_type_parentheses),
                            int(prefix_artist_with_by),
                        ),
                    )
                except sqlite3.IntegrityError as e:
                    raise CtaConflictError(f"Syntax profile '{name}' already exists") from e
                return CtaSyntaxProfileDTO(
                    id=profile_id,
                    name=name,
                    spaces_to_underscores=spaces_to_underscores,
                    escape_parentheses=escape_parentheses,
                    escape_colons=escape_colons,
                    append_type_parentheses=append_type_parentheses,
                    prefix_artist_with_by=prefix_artist_with_by,
                )
        finally:
            self._release_write()

    def update_syntax_profile(
        self,
        profile_id: str,
        name: str | None = None,
        spaces_to_underscores: bool | None = None,
        escape_parentheses: bool | None = None,
        escape_colons: bool | None = None,
        append_type_parentheses: bool | None = None,
        prefix_artist_with_by: bool | None = None,
    ) -> CtaSyntaxProfileDTO:
        self._assert_available()
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Syntax profile name must not be empty")
        self._acquire_write()
        try:
            with self._db.writer.transaction() as txn:
                txn.execute("SELECT id FROM cta_syntax_profiles WHERE id = ?", (profile_id,))
                if not txn.fetchone():
                    raise CtaNotFoundError(f"Syntax profile {profile_id} not found")

                updates: list[str] = []
                params: list[object] = []
                if name is not None:
                    updates.append("name = ?")
                    params.append(name)
                if spaces_to_underscores is not None:
                    updates.append("spaces_to_underscores = ?")
                    params.append(int(spaces_to_underscores))
                if escape_parentheses is not None:
                    updates.append("escape_parentheses = ?")
                    params.append(int(escape_parentheses))
                if escape_colons is not None:
                    updates.append("escape_colons = ?")
                    params.append(int(escape_colons))
                if append_type_parentheses is not None:
                    updates.append("append_type_parentheses = ?")
                    params.append(int(append_type_parentheses))
                if prefix_artist_with_by is not None:
                    updates.append("prefix_artist_with_by = ?")
                    params.append(int(prefix_artist_with_by))

                if updates:
                    params.append(profile_id)
                    try:
                        txn.execute(
                            f"UPDATE cta_syntax_profiles SET {', '.join(updates)} WHERE id = ?",
                            params,
                        )
                    except sqlite3.IntegrityError as e:
                        raise CtaConflictError(f"Syntax profile '{name}' already exists") from e

                txn.execute(
                    """SELECT id, name, spaces_to_underscores, escape_parentheses,
                              escape_colons, append_type_parentheses, prefix_artist_with_by
                       FROM cta_syntax_profiles WHERE id = ?""",
                    (profile_id,),
                )
                r = txn.fetchone()
                return CtaSyntaxProfileDTO(
                    id=r[0],
                    name=r[1],
                    spaces_to_underscores=bool(r[2]),
                    escape_parentheses=bool(r[3]),
                    escape_colons=bool(r[4]),
                    append_type_parentheses=bool(r[5]),
                    prefix_artist_with_by=bool(r[6]),
                )
        finally:
            self._release_write()

    def delete_syntax_profile(self, profile_id: str) -> None:
        self._assert_available()
        self._acquire_write()
        try:
            with self._db.writer.transaction() as txn:
                txn.execute("DELETE FROM cta_syntax_profiles WHERE id = ?", (profile_id,))
                if txn.rowcount == 0:
                    raise CtaNotFoundError(f"Syntax profile {profile_id} not found")
                # FK SET NULL handles cta_model_configs.syntax_profile_id
                self._cleanup_empty_model_configs(txn)
        finally:
            self._release_write()

    # ------------------------------------------------------------------
    # Model Config
    # ------------------------------------------------------------------

    def get_model_config(self, model_id: str) -> CtaModelConfigDTO:
        """Get CTA configuration for a model. Returns effective empty config when no row exists."""
        self._assert_available()
        with self._db.reader.transaction() as txn:
            txn.execute(
                "SELECT syntax_profile_id FROM cta_model_configs WHERE model_id = ?",
                (model_id,),
            )
            row = txn.fetchone()
            if not row:
                return CtaModelConfigDTO(model_id=model_id)

            syntax_profile_id = row[0]

            txn.execute(
                "SELECT tag_set_id FROM cta_model_tag_sets WHERE model_id = ?",
                (model_id,),
            )
            tag_set_ids = [r[0] for r in txn.fetchall()]

            return CtaModelConfigDTO(
                model_id=model_id,
                syntax_profile_id=syntax_profile_id,
                tag_set_ids=tag_set_ids,
            )

    def set_model_config(
        self,
        model_id: str,
        syntax_profile_id: str | None = None,
        tag_set_ids: list[str] | None = None,
    ) -> CtaModelConfigDTO:
        """Set CTA configuration for a model. Full replace in one transaction."""
        self._assert_available()
        self._acquire_write()
        try:
            with self._db.writer.transaction() as txn:
                normalized_tag_set_ids = list(dict.fromkeys(tag_set_ids or []))
                # Validate syntax profile exists when non-null
                if syntax_profile_id is not None:
                    txn.execute(
                        "SELECT id FROM cta_syntax_profiles WHERE id = ?",
                        (syntax_profile_id,),
                    )
                    if not txn.fetchone():
                        raise CtaNotFoundError(f"Syntax profile {syntax_profile_id} not found")

                # Validate all tag-set IDs exist
                if normalized_tag_set_ids:
                    for ts_id in normalized_tag_set_ids:
                        txn.execute(
                            "SELECT id FROM cta_tag_sets WHERE id = ?",
                            (ts_id,),
                        )
                        if not txn.fetchone():
                            raise CtaNotFoundError(f"Tag set {ts_id} not found")

                # When both profile is null and tag-set list is empty: delete config
                if syntax_profile_id is None and not normalized_tag_set_ids:
                    txn.execute("DELETE FROM cta_model_configs WHERE model_id = ?", (model_id,))
                    return CtaModelConfigDTO(model_id=model_id)

                # Upsert model config
                txn.execute(
                    """INSERT INTO cta_model_configs (model_id, syntax_profile_id)
                       VALUES (?, ?)
                       ON CONFLICT(model_id) DO UPDATE SET syntax_profile_id = excluded.syntax_profile_id""",
                    (model_id, syntax_profile_id),
                )

                # Replace model/tag-set junction rows
                txn.execute(
                    "DELETE FROM cta_model_tag_sets WHERE model_id = ?",
                    (model_id,),
                )
                if normalized_tag_set_ids:
                    for ts_id in normalized_tag_set_ids:
                        txn.execute(
                            "INSERT OR IGNORE INTO cta_model_tag_sets (model_id, tag_set_id) VALUES (?, ?)",
                            (model_id, ts_id),
                        )

                return CtaModelConfigDTO(
                    model_id=model_id,
                    syntax_profile_id=syntax_profile_id,
                    tag_set_ids=normalized_tag_set_ids,
                )
        finally:
            self._release_write()

    def request_model_config_cleanup(self, model_id: str) -> None:
        """Best-effort cleanup after model deletion. No-op if CTA unavailable or busy."""
        if not self.is_available:
            self._logger.debug("CTA unavailable, skipping model config cleanup")
            return

        if not self._write_lock.acquire(blocking=False):
            self._logger.debug("CTA write busy, deferring model config cleanup")
            with self._pending_cleanup_lock:
                self._pending_model_cleanup.add(model_id)
            return

        try:
            with self._db.writer.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM cta_model_configs WHERE model_id = ?",
                    (model_id,),
                )
        except Exception as e:
            self._logger.warning(f"CTA model config cleanup failed for {model_id}: {e}")
        finally:
            self._write_lock.release()

    def cleanup_orphan_model_configs(self, installed_model_ids: set[str]) -> int:
        """Remove model configs for models that no longer exist."""
        self._assert_available()
        self._acquire_write()
        try:
            return self._db.cleanup_orphan_model_configs(installed_model_ids)
        finally:
            self._release_write()

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def stage_import(
        self,
        owner_user_id: str,
        upload: BinaryIO,
    ) -> CtaImportStageDTO:
        """Stage a CSV upload for import."""
        self._assert_available()
        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_import import stage_upload

        session = self._import_manager.create_session(owner_user_id)
        try:
            return CtaImportStageDTO.model_validate(stage_upload(session, upload, self._logger))
        except Exception:
            self._import_manager.delete_session(session.id)
            raise

    def prepare_import(
        self,
        owner_user_id: str,
        session_id: str,
        tag_column: int = 0,
        popularity_column: int | None = None,
        type_column: int | None = None,
        delimiter: str | None = None,
        first_row_contains_column_names: bool | None = None,
    ) -> CtaImportPreviewDTO:
        """Prepare (full parse) a staged import."""
        self._assert_available()
        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_import import (
            prepare_import as do_prepare,
        )

        session = self._import_manager.get_session(session_id, owner_user_id)
        result = do_prepare(
            session,
            tag_column=tag_column,
            popularity_column=popularity_column,
            type_column=type_column,
            delimiter=delimiter,
            first_row_contains_column_names=first_row_contains_column_names,
            logger=self._logger,
        )
        return CtaImportPreviewDTO.model_validate(
            {
                "session_id": result.session_id,
                "state": result.state,
                "preview": result.preview,
                "summary": {
                    "rows_read": result.summary.rows_read,
                    "valid_rows": result.summary.valid_rows,
                    "skipped_rows": result.summary.skipped_rows,
                    "invalid_popularity_to_unknown": result.summary.invalid_popularity_to_unknown,
                    "unknown_types_to_other": result.summary.unknown_types_to_other,
                },
            }
        )

    def commit_import(
        self,
        owner_user_id: str,
        session_id: str,
        destination: dict[str, object],
    ) -> CtaImportResultDTO:
        """Commit a prepared import to the CTA database."""
        self._assert_available()
        self._acquire_write()
        try:
            from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_import import (
                CtaImportState,
            )
            from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_import import (
                commit_import as do_commit,
            )

            session = self._import_manager.get_session(session_id, owner_user_id)

            dest_type_value = destination.get("type", "uncategorized")
            dest_name_value = destination.get("name")
            dest_tag_set_id_value = destination.get("tag_set_id")
            import_mode_value = destination.get("mode", "merge")
            if not isinstance(dest_type_value, str) or not isinstance(import_mode_value, str):
                raise ValueError("Invalid import destination")
            if dest_name_value is not None and not isinstance(dest_name_value, str):
                raise ValueError("Invalid tag set name")
            if dest_tag_set_id_value is not None and not isinstance(dest_tag_set_id_value, str):
                raise ValueError("Invalid tag set ID")

            dest_type = dest_type_value
            dest_name = dest_name_value
            dest_tag_set_id = dest_tag_set_id_value
            import_mode = import_mode_value

            if dest_type == "new_set":
                dest_name = str(dest_name or "").strip()
                if not dest_name:
                    raise ValueError("A new tag set name is required")

            with self._db.writer.transaction() as txn:
                if dest_type == "new_set":
                    txn.execute("SELECT id FROM cta_tag_sets WHERE name = ?", (dest_name,))
                    if txn.fetchone() is not None:
                        raise CtaConflictError(f"Tag set '{dest_name}' already exists")
                result = do_commit(
                    session,
                    txn.connection,
                    destination_type=dest_type,
                    destination_name=dest_name,
                    destination_tag_set_id=dest_tag_set_id,
                    import_mode=import_mode,
                )

            session.state = CtaImportState.COMMITTED
            self._import_manager.delete_session(session_id)

            return CtaImportResultDTO(
                new_tags=result.new_tags,
                existing_tags_merged=result.existing_tags_merged,
                popularity_updated=result.popularity_updated,
                type_conflicts_ignored=result.type_conflicts_ignored,
                skipped_rows=result.skipped_rows,
            )
        finally:
            self._release_write()

    def cancel_import(self, owner_user_id: str, session_id: str) -> None:
        """Cancel and clean up an import session."""
        self._import_manager.get_session(session_id, owner_user_id)  # validates ownership
        self._import_manager.delete_session(session_id)
