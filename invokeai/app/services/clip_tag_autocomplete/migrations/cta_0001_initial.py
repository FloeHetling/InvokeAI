from __future__ import annotations

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


def build_migration() -> Migration:
    """Create the initial CTA schema."""

    def callback(cursor: sqlite3.Cursor) -> None:
        # Register CTA_CASEFOLD collation and function
        # These must be registered on every connection before schema use

        def cta_casefold(value: str | None) -> str:
            if value is None:
                return ""
            return value.casefold()

        def cta_casefold_collation(left: str, right: str) -> int:
            left_cf = left.casefold()
            right_cf = right.casefold()
            return (left_cf > right_cf) - (left_cf < right_cf)

        conn = cursor.connection
        conn.create_function("CTA_CASEFOLD", 1, cta_casefold, deterministic=True)
        conn.create_collation("CTA_CASEFOLD", cta_casefold_collation)

        # 1. cta_tags
        cursor.execute(
            """
            CREATE TABLE cta_tags (
                id TEXT NOT NULL PRIMARY KEY,
                canonical_content TEXT NOT NULL,
                popularity INTEGER,
                tag_type TEXT NOT NULL DEFAULT 'other',
                CHECK (popularity IS NULL OR popularity >= 0),
                CHECK (tag_type IN ('general', 'artist', 'copyright', 'character', 'meta', 'other'))
            )
            """
        )

        cursor.execute(
            """
            CREATE UNIQUE INDEX uq_cta_tags_semantic_identity
            ON cta_tags(canonical_content COLLATE CTA_CASEFOLD, tag_type)
            """
        )

        cursor.execute(
            """
            CREATE INDEX idx_cta_tags_type_content
            ON cta_tags(tag_type, canonical_content COLLATE CTA_CASEFOLD, id)
            """
        )

        # 2. cta_syntax_profiles
        cursor.execute(
            """
            CREATE TABLE cta_syntax_profiles (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                spaces_to_underscores INTEGER NOT NULL DEFAULT 1,
                escape_parentheses INTEGER NOT NULL DEFAULT 1,
                escape_colons INTEGER NOT NULL DEFAULT 1,
                append_type_parentheses INTEGER NOT NULL DEFAULT 0,
                prefix_artist_with_by INTEGER NOT NULL DEFAULT 0,
                CHECK (spaces_to_underscores IN (0, 1)),
                CHECK (escape_parentheses IN (0, 1)),
                CHECK (escape_colons IN (0, 1)),
                CHECK (append_type_parentheses IN (0, 1)),
                CHECK (prefix_artist_with_by IN (0, 1))
            )
            """
        )

        # 3. cta_tag_sets
        cursor.execute(
            """
            CREATE TABLE cta_tag_sets (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
            """
        )

        # 4. cta_model_configs
        cursor.execute(
            """
            CREATE TABLE cta_model_configs (
                model_id TEXT NOT NULL PRIMARY KEY,
                syntax_profile_id TEXT,
                FOREIGN KEY (syntax_profile_id)
                    REFERENCES cta_syntax_profiles(id)
                    ON DELETE SET NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX idx_cta_model_configs_syntax_profile_id
            ON cta_model_configs(syntax_profile_id, model_id)
            """
        )

        # 5. cta_tag_set_tags (many-to-many)
        cursor.execute(
            """
            CREATE TABLE cta_tag_set_tags (
                tag_set_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                PRIMARY KEY (tag_set_id, tag_id),
                FOREIGN KEY (tag_set_id)
                    REFERENCES cta_tag_sets(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (tag_id)
                    REFERENCES cta_tags(id)
                    ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX idx_cta_tag_set_tags_tag_id
            ON cta_tag_set_tags(tag_id, tag_set_id)
            """
        )

        # 6. cta_model_tag_sets (many-to-many)
        cursor.execute(
            """
            CREATE TABLE cta_model_tag_sets (
                model_id TEXT NOT NULL,
                tag_set_id TEXT NOT NULL,
                PRIMARY KEY (model_id, tag_set_id),
                FOREIGN KEY (model_id)
                    REFERENCES cta_model_configs(model_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (tag_set_id)
                    REFERENCES cta_tag_sets(id)
                    ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX idx_cta_model_tag_sets_tag_set_id
            ON cta_model_tag_sets(tag_set_id, model_id)
            """
        )

        # 7. FTS5 virtual table
        cursor.execute(
            """
            CREATE VIRTUAL TABLE cta_tags_fts USING fts5(
                tag_id UNINDEXED,
                search_content,
                tokenize = 'unicode61 remove_diacritics 0',
                prefix = '2 3'
            )
            """
        )

        # 8. FTS sync triggers
        cursor.execute(
            """
            CREATE TRIGGER cta_tags_fts_ai
            AFTER INSERT ON cta_tags
            BEGIN
                INSERT INTO cta_tags_fts(tag_id, search_content)
                VALUES (new.id, CTA_CASEFOLD(new.canonical_content));
            END
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER cta_tags_fts_ad
            AFTER DELETE ON cta_tags
            BEGIN
                DELETE FROM cta_tags_fts WHERE tag_id = old.id;
            END
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER cta_tags_fts_au
            AFTER UPDATE OF canonical_content ON cta_tags
            BEGIN
                DELETE FROM cta_tags_fts WHERE tag_id = old.id;
                INSERT INTO cta_tags_fts(tag_id, search_content)
                VALUES (new.id, CTA_CASEFOLD(new.canonical_content));
            END
            """
        )

    return Migration(
        id="cta_0001_initial",
        depends_on=None,
        callback=callback,
    )
