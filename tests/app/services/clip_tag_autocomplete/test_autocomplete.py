"""Backend autocomplete contract test for CTA.

Uses a real CTA database with FTS5 to verify search/ranking behavior.
"""

from __future__ import annotations

import sqlite3
import uuid
from io import BytesIO

import pytest

from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import (
    CtaBulkTagRequest,
    CtaConflictError,
    CtaNotFoundError,
    CtaSyntaxProfileDTO,
    CtaWriteBusyError,
    render_cta_tag,
)
from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_database import CtaDatabase
from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_search import (
    build_fts_phrase,
    classify_text_rank,
    normalize_autocomplete_query,
)
from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_service import (
    ClipTagAutocompleteService,
)
from invokeai.app.services.config.config_default import InvokeAIAppConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cta_config(tmp_path):
    """Create a minimal config pointing to a temp directory."""
    config = InvokeAIAppConfig()
    config.db_dir = tmp_path
    config.use_memory_db = False
    return config


@pytest.fixture
def cta_service(cta_config, caplog):
    """Create a CTA service with a fresh database."""
    from logging import getLogger

    logger = getLogger("test_cta")
    db = CtaDatabase(config=cta_config, logger=logger)
    db.initialize()
    assert db.is_available, f"CTA init failed: {db.status}"
    return ClipTagAutocompleteService(cta_db=db, logger=logger)


@pytest.fixture
def populated_service(cta_service):
    """Populate the CTA database with test tags for autocomplete testing."""
    # Insert test tags
    tags = [
        ("beholder", 50000, "character"),
        ("red bedding", 30000, "general"),
        ("red bed", 25000, "general"),
        ("applejack", 12000, "character"),
        ("applejack", 5000, "artist"),
        ("apple pie", 8000, "general"),
        ("alpenliebe", 1000, "general"),
        ("be happy", 2000, "general"),
        ("trespassers begone", 500, "general"),
        ("sally acorn", 18432, "character"),
        ("sally acorn", 2401, "artist"),
        ("pinkie pie", 15000, "character"),
        ("rarity", 14000, "character"),
        ("fluttershy", 13000, "character"),
        ("rainbow dash", 16000, "character"),
        ("twilight sparkle", 17000, "character"),
    ]

    with cta_service._db.writer.transaction() as txn:
        for canonical, pop, tag_type in tags:
            txn.execute(
                """INSERT INTO cta_tags (id, canonical_content, popularity, tag_type)
                   VALUES (?, ?, ?, ?)""",
                (str(uuid.uuid4()), canonical, pop, tag_type),
            )

    # Create a tag set and associate some tags
    with cta_service._db.writer.transaction() as txn:
        tag_set_id = str(uuid.uuid4())
        txn.execute(
            "INSERT INTO cta_tag_sets (id, name) VALUES (?, ?)",
            (tag_set_id, "My Little Pony"),
        )

        # Associate applejack (character) with the tag set
        txn.execute("SELECT id FROM cta_tags WHERE canonical_content = 'applejack' AND tag_type = 'character'")
        applejack_id = txn.fetchone()[0]
        txn.execute(
            "INSERT INTO cta_tag_set_tags (tag_set_id, tag_id) VALUES (?, ?)",
            (tag_set_id, applejack_id),
        )

        # Create a model config with this tag set
        model_id = "test-model-key"
        txn.execute(
            "INSERT INTO cta_model_configs (model_id) VALUES (?)",
            (model_id,),
        )
        txn.execute(
            "INSERT INTO cta_model_tag_sets (model_id, tag_set_id) VALUES (?, ?)",
            (model_id, tag_set_id),
        )

    return cta_service


# ---------------------------------------------------------------------------
# Search helper tests
# ---------------------------------------------------------------------------


class TestNormalizeQuery:
    def test_underscores_to_spaces(self):
        assert normalize_autocomplete_query("pippi_long") == "pippi long"

    def test_multiple_underscores(self):
        assert normalize_autocomplete_query("apple__pie") == "apple pie"

    def test_trailing_underscore(self):
        assert normalize_autocomplete_query("apple_") == "apple"

    def test_case_insensitive(self):
        assert normalize_autocomplete_query("Red") == "red"

    def test_collapse_whitespace(self):
        assert normalize_autocomplete_query("  red   bed  ") == "red bed"


class TestBuildFtsPhrase:
    def test_simple_query(self):
        assert build_fts_phrase("red") == '"red" *'

    def test_multiword_query(self):
        assert build_fts_phrase("red bed") == '"red bed" *'

    def test_escapes_quotes(self):
        assert build_fts_phrase('foo"bar') == '"foo""bar" *'


class TestClassifyTextRank:
    def test_exact_match(self):
        assert classify_text_rank("red", "red") == 0

    def test_canonical_start(self):
        assert classify_text_rank("red bedding", "red") == 1

    def test_later_word(self):
        assert classify_text_rank("red bedding", "bed") == 2

    def test_case_insensitive(self):
        assert classify_text_rank("Red Bedding", "red") == 1


# ---------------------------------------------------------------------------
# Autocomplete contract tests
# ---------------------------------------------------------------------------


class TestAutocomplete:
    def test_empty_query_returns_empty(self, populated_service):
        result = populated_service.autocomplete("", None)
        assert result == []

    def test_single_char_returns_empty(self, populated_service):
        result = populated_service.autocomplete("a", None)
        assert result == []

    def test_word_prefix_match(self, populated_service):
        results = populated_service.autocomplete("be", None)
        names = [r.canonical_content for r in results]
        assert "beholder" in names
        assert "red bedding" in names
        assert "be happy" in names

    def test_no_inside_word_match(self, populated_service):
        results = populated_service.autocomplete("be", None)
        names = [r.canonical_content for r in results]
        assert "alpenliebe" not in names

    def test_exact_ranked_first(self, populated_service):
        results = populated_service.autocomplete("applejack", None)
        assert len(results) >= 2
        # applejack (character and artist) should be first
        assert results[0].canonical_content == "applejack"
        assert results[1].canonical_content == "applejack"

    def test_canonical_start_ranked_before_later_word(self, populated_service):
        results = populated_service.autocomplete("red", None)
        names = [r.canonical_content for r in results]
        # "red bedding" and "red bed" start with "red" — should come before later-word matches
        assert names.index("red bedding") < len(names)
        assert names.index("red bed") < len(names)

    def test_model_set_boost(self, populated_service):
        # applejack (character) is in the tag set associated with the model
        with populated_service._db.writer.transaction() as txn:
            txn.execute(
                "UPDATE cta_tags SET popularity = 100000 WHERE canonical_content = 'applejack' AND tag_type = 'artist'"
            )
        results = populated_service.autocomplete("applejack", "test-model-key")
        assert results[0].tag_type == "character"
        assert results[1].tag_type == "artist"

    def test_model_boost_does_not_beat_better_text_rank(self, populated_service):
        with populated_service._db.writer.transaction() as txn:
            txn.execute("SELECT tag_set_id FROM cta_model_tag_sets WHERE model_id = 'test-model-key'")
            tag_set_id = txn.fetchone()[0]
            txn.execute("SELECT id FROM cta_tags WHERE canonical_content = 'red bedding'")
            later_match_id = txn.fetchone()[0]
            txn.execute(
                "INSERT INTO cta_tag_set_tags (tag_set_id, tag_id) VALUES (?, ?)",
                (tag_set_id, later_match_id),
            )
        results = populated_service.autocomplete("be", "test-model-key")
        names = [result.canonical_content for result in results]
        assert names.index("beholder") < names.index("red bedding")

    def test_unknown_popularity_sorts_last_within_rank(self, populated_service):
        with populated_service._db.writer.transaction() as txn:
            txn.execute(
                "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, ?, NULL, 'general')",
                (str(uuid.uuid4()), "apple unknown"),
            )
        results = populated_service.autocomplete("apple", None)
        assert results[-1].canonical_content == "apple unknown"

    def test_punctuation_only_query_is_safe(self, populated_service):
        assert populated_service.autocomplete("~~", None) == []

    def test_limit_20(self, populated_service):
        # Insert many tags to test LIMIT
        with populated_service._db.writer.transaction() as txn:
            for i in range(25):
                txn.execute(
                    """INSERT INTO cta_tags (id, canonical_content, popularity, tag_type)
                       VALUES (?, ?, ?, ?)""",
                    (str(uuid.uuid4()), f"test tag {i:03d}", 100, "general"),
                )
        results = populated_service.autocomplete("test", None)
        assert len(results) == 20

    def test_category_filters_are_disjoint(self, populated_service):
        with populated_service._db.writer.transaction() as txn:
            for canonical, tag_type in [
                ("apple studio", "copyright"),
                ("apple metadata", "meta"),
                ("apple miscellaneous", "other"),
            ]:
                txn.execute(
                    """INSERT INTO cta_tags (id, canonical_content, popularity, tag_type)
                       VALUES (?, ?, ?, ?)""",
                    (str(uuid.uuid4()), canonical, 100, tag_type),
                )

        artist_results = populated_service.autocomplete("apple", None, "artist")
        character_results = populated_service.autocomplete("apple", None, "character")
        other_results = populated_service.autocomplete("apple", None, "other")

        assert {result.tag_type for result in artist_results} == {"artist", "copyright"}
        assert {result.tag_type for result in character_results} == {"character"}
        assert {result.tag_type for result in other_results} <= {"general", "meta", "other"}
        assert {"general", "meta", "other"} <= {result.tag_type for result in other_results}

    def test_filtered_results_page_in_batches_of_50(self, populated_service):
        with populated_service._db.writer.transaction() as txn:
            for i in range(55):
                txn.execute(
                    """INSERT INTO cta_tags (id, canonical_content, popularity, tag_type)
                       VALUES (?, ?, ?, 'general')""",
                    (str(uuid.uuid4()), f"browse tag {i:03d}", 1_000 - i),
                )

        first_page = populated_service.autocomplete("browse", None, "other", 0, 50)
        second_page = populated_service.autocomplete("browse", None, "other", 50, 50)

        assert len(first_page) == 50
        assert len(second_page) == 5
        assert {result.id for result in first_page}.isdisjoint(result.id for result in second_page)

    def test_rendered_content_follows_profile(self, populated_service):
        profile = CtaSyntaxProfileDTO(
            id="test",
            name="test",
            spaces_to_underscores=True,
            escape_parentheses=True,
            escape_colons=True,
            append_type_parentheses=True,
            prefix_artist_with_by=False,
        )
        # applejack (character) should render with type parentheses
        with populated_service._db.reader.transaction() as txn:
            txn.execute(
                "SELECT id, canonical_content, tag_type FROM cta_tags WHERE canonical_content = 'applejack' AND tag_type = 'character'"
            )
            row = txn.fetchone()
            rendered = render_cta_tag(row[1], row[2], profile)
            assert "\\(character\\)" in rendered
            assert "_" in rendered  # spaces to underscores

    def test_artist_prefix_stays_space_separated_from_underscored_content(self):
        profile = CtaSyntaxProfileDTO(
            id="artist-prefix",
            name="artist-prefix",
            spaces_to_underscores=True,
            prefix_artist_with_by=True,
        )

        assert render_cta_tag("marcie montis", "artist", profile) == "by marcie_montis"
        # A canonical value that already carries the textual prefix must not become double-prefixed.
        assert render_cta_tag("by marcie montis", "artist", profile) == "by marcie_montis"
        # Non-artist tags keep the existing spaces-to-underscores behavior.
        assert render_cta_tag("marcie montis", "general", profile) == "marcie_montis"

    def test_service_applies_model_syntax_profile(self, populated_service):
        profile = populated_service.create_syntax_profile(
            "Prompt syntax",
            append_type_parentheses=True,
            prefix_artist_with_by=True,
        )
        populated_service.set_model_config("render-model", syntax_profile_id=profile.id)
        results = populated_service.autocomplete("applejack", "render-model")
        artist = next(result for result in results if result.tag_type == "artist")
        assert artist.rendered_content == "by applejack_\\(artist\\)"


# ---------------------------------------------------------------------------
# Popularity merge test
# ---------------------------------------------------------------------------


class TestPopularityMerge:
    def test_both_none(self):
        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import merge_popularity

        assert merge_popularity(None, None) is None

    def test_existing_none(self):
        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import merge_popularity

        assert merge_popularity(None, 100) == 100

    def test_incoming_none(self):
        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import merge_popularity

        assert merge_popularity(100, None) == 100

    def test_max(self):
        from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import merge_popularity

        assert merge_popularity(100, 200) == 200
        assert merge_popularity(200, 100) == 200


class TestManagerMutations:
    def test_pagination_keeps_filtered_total_count(self, cta_service):
        with cta_service._db.writer.transaction() as txn:
            for index in range(4):
                txn.execute(
                    "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, ?, ?, 'general')",
                    (str(uuid.uuid4()), f"page tag {index}", index),
                )
        first = cta_service.list_tags(q="page", limit=2)
        second = cta_service.list_tags(q="page", cursor=first.next_cursor, limit=2)
        assert first.total_count == 4
        assert second.total_count == 4
        assert {item.id for item in first.items}.isdisjoint(item.id for item in second.items)

    def test_manual_collision_merges_memberships_and_preserves_target_id(self, cta_service):
        first_set = cta_service.create_tag_set("First")
        second_set = cta_service.create_tag_set("Second")
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        with cta_service._db.writer.transaction() as txn:
            txn.execute(
                "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, 'source', 20, 'character')",
                (source_id,),
            )
            txn.execute(
                "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, 'target', 10, 'character')",
                (target_id,),
            )
            txn.execute(
                "INSERT INTO cta_tag_set_tags (tag_set_id, tag_id) VALUES (?, ?), (?, ?)",
                (first_set.id, source_id, second_set.id, target_id),
            )

        result = cta_service.update_tag(source_id, canonical_content="target")

        assert result.merged is True
        assert result.tag.id == target_id
        assert result.tag.popularity == 20
        assert set(result.tag.tag_set_ids) == {first_set.id, second_set.id}
        with pytest.raises(CtaNotFoundError):
            cta_service.get_tag(source_id)

    def test_bulk_set_type_uses_snapshot_and_merges_collision(self, cta_service):
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        with cta_service._db.writer.transaction() as txn:
            txn.execute(
                "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, 'bulk target', 5, 'other')",
                (source_id,),
            )
            txn.execute(
                "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, 'bulk target', 10, 'character')",
                (target_id,),
            )
        request = CtaBulkTagRequest.model_validate(
            {
                "selection": {"mode": "filter", "filter": {"q": "bulk", "tag_type": "other"}},
                "operation": {"type": "set_type", "tag_type": "character"},
            }
        )

        result = cta_service.bulk_mutate_tags(request)

        assert result.selected_count == 1
        assert result.affected_count == 1
        assert result.merged_count == 1
        assert cta_service.get_tag(target_id).popularity == 10
        with pytest.raises(CtaNotFoundError):
            cta_service.get_tag(source_id)

    def test_busy_writer_fails_fast(self, cta_service):
        cta_service._write_lock.acquire()
        try:
            with pytest.raises(CtaWriteBusyError):
                cta_service.create_tag_set("Blocked")
        finally:
            cta_service._write_lock.release()

    def test_missing_deletes_raise_not_found(self, cta_service):
        with pytest.raises(CtaNotFoundError):
            cta_service.delete_tag("missing")
        with pytest.raises(CtaNotFoundError):
            cta_service.delete_tag_set("missing")
        with pytest.raises(CtaNotFoundError):
            cta_service.delete_syntax_profile("missing")


class TestImportWorkflow:
    def test_single_column_import_keeps_first_row(self, cta_service):
        staged = cta_service.stage_import("owner", BytesIO(b"first_tag\nsecond_tag\n"))
        assert staged.detected_header is False
        assert staged.is_single_column is True
        prepared = cta_service.prepare_import(
            "owner",
            staged.session_id,
            delimiter=staged.detected_delimiter,
            first_row_contains_column_names=False,
        )
        assert prepared.summary.rows_read == 2
        assert prepared.summary.valid_rows == 2
        result = cta_service.commit_import("owner", staged.session_id, {"type": "uncategorized"})
        assert result.new_tags == 2
        assert {item.canonical_content for item in cta_service.list_tags(limit=50).items} == {
            "first tag",
            "second tag",
        }

    def test_import_upgrades_other_in_place_and_preserves_membership(self, cta_service):
        original_set = cta_service.create_tag_set("Original")
        destination_set = cta_service.create_tag_set("Destination")
        original_id = str(uuid.uuid4())
        with cta_service._db.writer.transaction() as txn:
            txn.execute(
                "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, 'sally acorn', 10, 'other')",
                (original_id,),
            )
            txn.execute(
                "INSERT INTO cta_tag_set_tags (tag_set_id, tag_id) VALUES (?, ?)",
                (original_set.id, original_id),
            )
        staged = cta_service.stage_import(
            "owner",
            BytesIO(b"tag,popularity,type\nsally_acorn,20,character\n"),
        )
        cta_service.prepare_import(
            "owner",
            staged.session_id,
            tag_column=0,
            popularity_column=1,
            type_column=2,
            delimiter=",",
            first_row_contains_column_names=True,
        )
        result = cta_service.commit_import(
            "owner",
            staged.session_id,
            {"type": "existing_set", "tag_set_id": destination_set.id, "mode": "merge"},
        )
        tag = cta_service.get_tag(original_id)
        assert result.existing_tags_merged == 1
        assert tag.tag_type == "character"
        assert tag.popularity == 20
        assert set(tag.tag_set_ids) == {original_set.id, destination_set.id}

    def test_import_artist_and_character_coexist(self, cta_service):
        character_id = str(uuid.uuid4())
        with cta_service._db.writer.transaction() as txn:
            txn.execute(
                "INSERT INTO cta_tags (id, canonical_content, popularity, tag_type) VALUES (?, 'shared name', 10, 'character')",
                (character_id,),
            )
        staged = cta_service.stage_import("owner", BytesIO(b"tag,type\nshared_name,artist\n"))
        cta_service.prepare_import(
            "owner",
            staged.session_id,
            tag_column=0,
            type_column=1,
            delimiter=",",
            first_row_contains_column_names=True,
        )
        cta_service.commit_import("owner", staged.session_id, {"type": "uncategorized"})
        tags = cta_service.list_tags(q="shared", limit=50).items
        assert {(tag.canonical_content, tag.tag_type) for tag in tags} == {
            ("shared name", "artist"),
            ("shared name", "character"),
        }

    def test_new_set_collision_rolls_back_import(self, cta_service):
        cta_service.create_tag_set("Existing")
        staged = cta_service.stage_import("owner", BytesIO(b"new_tag\n"))
        cta_service.prepare_import(
            "owner",
            staged.session_id,
            delimiter=",",
            first_row_contains_column_names=False,
        )
        with pytest.raises(CtaConflictError):
            cta_service.commit_import(
                "owner",
                staged.session_id,
                {"type": "new_set", "name": " Existing "},
            )
        assert cta_service.list_tags(limit=50).total_count == 0


def test_unknown_sidecar_migration_is_preserved(cta_config, caplog):
    cta_path = cta_config.db_path.parent / "cta.db"
    with sqlite3.connect(cta_path) as conn:
        conn.execute("CREATE TABLE applied_migrations (migration_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO applied_migrations VALUES ('cta_0001_initial')")
        conn.execute("INSERT INTO applied_migrations VALUES ('cta_9999_future')")
        conn.execute("CREATE TABLE future_data (value TEXT)")
        conn.execute("INSERT INTO future_data VALUES ('preserve me')")
    original_bytes = cta_path.read_bytes()

    from logging import getLogger

    db = CtaDatabase(config=cta_config, logger=getLogger("test_cta_incompatible"))
    db.initialize()

    assert db.is_available is False
    assert db.status.reason == "database_incompatible"
    assert cta_path.read_bytes() == original_bytes
