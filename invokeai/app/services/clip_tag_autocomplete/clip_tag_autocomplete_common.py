"""CLIP Tag Autocomplete — domain types, exceptions, and pure helpers.

Contains only DTOs (Pydantic models), exception classes, and stateless
helper functions. No database, no I/O, no framework dependencies.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Tag type literal
# ---------------------------------------------------------------------------

CtaTagType = Literal[
    "general",
    "artist",
    "copyright",
    "character",
    "meta",
    "other",
]
CtaAutocompleteFilter = Literal["artist", "character", "other"]

CTA_VALID_TAG_TYPES: set[str] = {
    "general",
    "artist",
    "copyright",
    "character",
    "meta",
    "other",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CtaStatusReason(str, Enum):
    FTS5_UNAVAILABLE = "fts5_unavailable"
    DATABASE_INCOMPATIBLE = "database_incompatible"
    DATABASE_ERROR = "database_error"


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


class CtaStatus(BaseModel):
    available: bool
    reason: CtaStatusReason | None = None


class CtaTagDTO(BaseModel):
    id: str
    canonical_content: str
    popularity: int | None = None
    tag_type: CtaTagType


class CtaAutocompleteCandidate(CtaTagDTO):
    rendered_content: str


class CtaSyntaxProfileDTO(BaseModel):
    id: str
    name: str
    spaces_to_underscores: bool = True
    escape_parentheses: bool = True
    escape_colons: bool = True
    append_type_parentheses: bool = False
    prefix_artist_with_by: bool = False


class CtaTagSetDTO(BaseModel):
    id: str
    name: str
    tag_count: int = 0
    model_count: int = 0


class CtaTagPage(BaseModel):
    items: list[CtaTagDTO]
    next_cursor: str | None = None
    total_count: int


class CtaTagDetailDTO(CtaTagDTO):
    tag_set_ids: list[str] = Field(default_factory=list)


class CtaTagMutationResult(BaseModel):
    tag: CtaTagDetailDTO
    merged: bool = False


class CtaModelConfigDTO(BaseModel):
    model_id: str
    syntax_profile_id: str | None = None
    tag_set_ids: list[str] = Field(default_factory=list)


class CtaTagUpdate(BaseModel):
    canonical_content: str | None = Field(default=None, min_length=1, max_length=2048)
    tag_type: CtaTagType | None = None
    popularity: int | None = Field(default=None, ge=0)
    tag_set_ids: list[str] | None = None


class CtaTagSetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class CtaTagSetUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class CtaSyntaxProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    spaces_to_underscores: bool = True
    escape_parentheses: bool = True
    escape_colons: bool = True
    append_type_parentheses: bool = False
    prefix_artist_with_by: bool = False


class CtaSyntaxProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    spaces_to_underscores: bool | None = None
    escape_parentheses: bool | None = None
    escape_colons: bool | None = None
    append_type_parentheses: bool | None = None
    prefix_artist_with_by: bool | None = None


class CtaModelConfigUpdate(BaseModel):
    syntax_profile_id: str | None = None
    tag_set_ids: list[str] = Field(default_factory=list)


class CtaTagFilter(BaseModel):
    q: str | None = Field(default=None, max_length=512)
    tag_type: CtaTagType | None = None
    tag_set_id: str | None = None
    uncategorized: bool = False

    @model_validator(mode="after")
    def validate_membership_filter(self) -> "CtaTagFilter":
        if self.tag_set_id is not None and self.uncategorized:
            raise ValueError("tag_set_id and uncategorized are mutually exclusive")
        return self


class CtaBulkIdSelection(BaseModel):
    mode: Literal["ids"]
    ids: list[str] = Field(min_length=1, max_length=1000)


class CtaBulkFilterSelection(BaseModel):
    mode: Literal["filter"]
    filter: CtaTagFilter


CtaBulkSelection = Annotated[CtaBulkIdSelection | CtaBulkFilterSelection, Field(discriminator="mode")]


class CtaBulkAddToSet(BaseModel):
    type: Literal["add_to_set"]
    tag_set_id: str


class CtaBulkRemoveFromSet(BaseModel):
    type: Literal["remove_from_set"]
    tag_set_id: str


class CtaBulkSetType(BaseModel):
    type: Literal["set_type"]
    tag_type: CtaTagType


class CtaBulkDelete(BaseModel):
    type: Literal["delete"]


CtaBulkOperation = Annotated[
    CtaBulkAddToSet | CtaBulkRemoveFromSet | CtaBulkSetType | CtaBulkDelete,
    Field(discriminator="type"),
]


class CtaBulkTagRequest(BaseModel):
    selection: CtaBulkSelection
    operation: CtaBulkOperation


class CtaBulkResult(BaseModel):
    selected_count: int
    affected_count: int
    merged_count: int = 0


CtaImportDelimiter = Literal[",", ";", "\t", "|"]


class CtaImportMapping(BaseModel):
    delimiter: CtaImportDelimiter
    first_row_contains_column_names: bool
    tag_column: int = Field(ge=0)
    popularity_column: int | None = Field(default=None, ge=0)
    type_column: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_distinct_columns(self) -> "CtaImportMapping":
        mapped = [self.tag_column]
        if self.popularity_column is not None:
            mapped.append(self.popularity_column)
        if self.type_column is not None:
            mapped.append(self.type_column)
        if len(mapped) != len(set(mapped)):
            raise ValueError("Each import field must map to a different column")
        return self


class CtaImportUncategorizedDestination(BaseModel):
    type: Literal["uncategorized"]


class CtaImportNewSetDestination(BaseModel):
    type: Literal["new_set"]
    name: str = Field(min_length=1, max_length=255)


class CtaImportExistingSetDestination(BaseModel):
    type: Literal["existing_set"]
    tag_set_id: str
    mode: Literal["merge", "replace"] = "merge"


CtaImportDestination = Annotated[
    CtaImportUncategorizedDestination | CtaImportNewSetDestination | CtaImportExistingSetDestination,
    Field(discriminator="type"),
]


class CtaImportCommitRequest(BaseModel):
    destination: CtaImportDestination


class CtaImportStageDTO(BaseModel):
    session_id: str
    detected_delimiter: CtaImportDelimiter
    detected_header: bool
    columns: list[str]
    sample_rows: list[list[str]]
    is_single_column: bool


class CtaImportSummaryDTO(BaseModel):
    rows_read: int
    valid_rows: int
    skipped_rows: int
    invalid_popularity_to_unknown: int
    unknown_types_to_other: int


class CtaImportPreviewRow(BaseModel):
    canonical_content: str
    popularity: int | None = None
    tag_type: CtaTagType


class CtaImportPreviewDTO(BaseModel):
    session_id: str
    state: Literal["prepared"]
    preview: list[CtaImportPreviewRow]
    summary: CtaImportSummaryDTO


class CtaImportResultDTO(BaseModel):
    new_tags: int
    existing_tags_merged: int
    popularity_updated: int
    type_conflicts_ignored: int
    skipped_rows: int


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CtaUnavailableError(RuntimeError): ...


class CtaNotFoundError(LookupError): ...


class CtaConflictError(ValueError): ...


class CtaWriteBusyError(RuntimeError): ...


class CtaImportSessionNotFoundError(LookupError): ...


class CtaImportFatalError(ValueError): ...


class CtaModelNotFoundError(LookupError): ...


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def merge_popularity(existing: int | None, incoming: int | None) -> int | None:
    """Standard popularity merge: max(existing, incoming) with NULL semantics."""
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    return max(existing, incoming)


def render_cta_tag(
    canonical_content: str,
    tag_type: str,
    profile: CtaSyntaxProfileDTO | None,
) -> str:
    """Render canonical content using a syntax profile. Pure function."""
    if profile is None:
        return canonical_content

    value = canonical_content
    should_prefix_artist = profile.prefix_artist_with_by and tag_type == "artist"

    # Treat a canonical leading "by " as the same syntax decoration so rendering stays idempotent.
    if should_prefix_artist and value.startswith("by "):
        value = value[3:]

    # 1. Append type parentheses if required
    if profile.append_type_parentheses and tag_type != "other":
        # Idempotent: check if content already has a separate (type) group
        groups = re.findall(r"\(([^()]*)\)", value)
        if tag_type not in groups:
            value = f"{value} ({tag_type})"

    # 2. Convert spaces inside the rendered tag body to underscores.
    if profile.spaces_to_underscores:
        value = value.replace(" ", "_")

    # 3. Add syntax decoration after body transforms so "by " keeps its literal separator.
    if should_prefix_artist:
        value = f"by {value}"

    # 4. Escape parentheses
    if profile.escape_parentheses:
        value = value.replace("(", "\\(").replace(")", "\\)")

    # 5. Escape colons
    if profile.escape_colons:
        value = value.replace(":", "\\:")

    return value


def normalize_import_content(raw: str) -> str:
    """Normalize imported tag content to CTA canonical content.

    Pipeline: trim -> _ to space -> collapse whitespace -> unescape \\( \\) \\: -> trim
    """
    value = raw.strip()
    value = value.replace("_", " ")
    value = " ".join(value.split())  # collapse whitespace
    value = value.replace("\\(", "(")
    value = value.replace("\\)", ")")
    value = value.replace("\\:", ":")
    return value.strip()
