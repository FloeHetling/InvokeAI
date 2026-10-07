"""CLIP Tag Autocomplete — search helpers, FTS query builder, and cursor pagination."""

from __future__ import annotations

import base64
import json
from typing import Any, cast

# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------


def normalize_autocomplete_query(raw: str) -> str:
    """Normalize a raw autocomplete query for FTS search.

    Pipeline: _ to space -> collapse whitespace -> trim -> casefold.
    """
    value = raw.replace("_", " ")
    value = " ".join(value.split())  # collapse whitespace
    return value.strip().casefold()


# ---------------------------------------------------------------------------
# FTS5 expression builder
# ---------------------------------------------------------------------------


def build_fts_phrase(query: str) -> str:
    """Build a safe FTS5 phrase query from a normalized query.

    Wraps the query in double quotes (escaping internal quotes) and appends
    the FTS5 prefix marker '*' to the final term. This ensures:
    - No user input is interpreted as FTS5 query syntax
    - The query matches at word boundaries (via unicode61 tokenizer)
    - The final token is prefix-matched

    Examples:
        "red"           -> "red" *
        "red bed"       -> "red bed" *
        'foo"bar'       -> "foo""bar" *
    """
    escaped = query.replace('"', '""')
    return f'"{escaped}" *'


# ---------------------------------------------------------------------------
# Ranking classification
# ---------------------------------------------------------------------------


def classify_text_rank(canonical_content: str, normalized_query: str) -> int:
    """Classify the text match quality for ranking.

    Returns:
        0 = exact canonical match
        1 = match at beginning of complete canonical tag
        2 = match at beginning of a later word

    The FTS candidate set already guarantees a valid token-boundary phrase
    match, so we can safely use simple string operations for classification.
    """
    casefolded = canonical_content.casefold()

    if casefolded == normalized_query:
        return 0  # exact

    if casefolded.startswith(normalized_query):
        return 1  # full canonical start

    # Check if query matches at a word boundary (after space, punctuation, or start)
    # Since FTS already validated word-boundary match, we just need to classify
    # "later word start" as rank 2
    return 2


# ---------------------------------------------------------------------------
# Cursor/keyset pagination
# ---------------------------------------------------------------------------


def encode_cursor(data: dict[str, Any]) -> str:
    """Encode a cursor dict to an opaque base64url string."""
    json_bytes = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(json_bytes).decode("ascii")


def decode_cursor(cursor: str) -> dict[str, Any]:
    """Decode an opaque base64url cursor string back to a dict."""
    try:
        json_bytes = base64.urlsafe_b64decode(cursor.encode("ascii"))
        decoded = json.loads(json_bytes.decode("utf-8"))
        if not isinstance(decoded, dict) or not all(
            isinstance(decoded.get(field), str) for field in ("canonical_content", "tag_type", "id")
        ):
            raise ValueError("Cursor payload is invalid")
        return cast(dict[str, Any], decoded)
    except Exception as e:
        raise ValueError(f"Invalid cursor: {e}")


def build_keyset_condition(
    cursor_data: dict[str, Any],
    table_alias: str = "t",
) -> tuple[str, dict[str, Any]]:
    """Build a SQL WHERE condition for keyset/cursor pagination.

    Returns (condition_sql, params) for ascending ordering:
    canonical_content COLLATE CTA_CASEFOLD > :cursor_content
    OR (content = :cursor_content AND tag_type > :cursor_type)
    OR (content = :cursor_content AND tag_type = :cursor_type AND id > :cursor_id)
    """
    prefix = f"{table_alias}." if table_alias else ""
    params: dict[str, Any] = {
        "cursor_content": cursor_data["canonical_content"],
        "cursor_type": cursor_data["tag_type"],
        "cursor_id": cursor_data["id"],
    }

    condition = f"""
        ({prefix}canonical_content COLLATE CTA_CASEFOLD > :cursor_content)
        OR (
            {prefix}canonical_content COLLATE CTA_CASEFOLD = :cursor_content
            AND {prefix}tag_type > :cursor_type
        )
        OR (
            {prefix}canonical_content COLLATE CTA_CASEFOLD = :cursor_content
            AND {prefix}tag_type = :cursor_type
            AND {prefix}id > :cursor_id
        )
    """

    return condition.strip(), params
