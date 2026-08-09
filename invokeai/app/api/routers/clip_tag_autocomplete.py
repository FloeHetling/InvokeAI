from __future__ import annotations

from typing import Annotated, NoReturn

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_common import (
    CtaAutocompleteCandidate,
    CtaAutocompleteFilter,
    CtaBulkResult,
    CtaBulkTagRequest,
    CtaConflictError,
    CtaImportCommitRequest,
    CtaImportMapping,
    CtaImportPreviewDTO,
    CtaImportResultDTO,
    CtaImportStageDTO,
    CtaModelConfigDTO,
    CtaModelConfigUpdate,
    CtaStatus,
    CtaSyntaxProfileCreate,
    CtaSyntaxProfileDTO,
    CtaSyntaxProfileUpdate,
    CtaTagDetailDTO,
    CtaTagMutationResult,
    CtaTagPage,
    CtaTagSetCreate,
    CtaTagSetDTO,
    CtaTagSetUpdate,
    CtaTagType,
    CtaTagUpdate,
    CtaUnavailableError,
    CtaWriteBusyError,
)
from invokeai.app.services.clip_tag_autocomplete.clip_tag_autocomplete_service import ClipTagAutocompleteService

clip_tag_autocomplete_router = APIRouter(
    prefix="/v1/clip_tag_autocomplete",
    tags=["clip_tag_autocomplete"],
)


def _get_service() -> ClipTagAutocompleteService:
    service = ApiDependencies.invoker.services.clip_tag_autocomplete
    if service is None or not service.is_available:
        raise HTTPException(status_code=503, detail="CTA service not available")
    return service


def _raise_cta_error(error: Exception) -> NoReturn:
    if isinstance(error, CtaUnavailableError):
        raise HTTPException(status_code=503, detail="CTA service not available") from error
    if isinstance(error, CtaWriteBusyError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, CtaConflictError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, LookupError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error)) from error
    raise error


@clip_tag_autocomplete_router.get(
    "/status",
    operation_id="get_cta_status",
    response_model=CtaStatus,
)
async def get_cta_status(
    current_user: CurrentUserOrDefault,
) -> CtaStatus:
    """Returns the CTA availability status."""
    service = ApiDependencies.invoker.services.clip_tag_autocomplete
    if service is None:
        return CtaStatus(available=False, reason=None)
    return await run_in_threadpool(lambda: service.status)


@clip_tag_autocomplete_router.get(
    "/autocomplete",
    operation_id="cta_autocomplete",
    response_model=list[CtaAutocompleteCandidate],
)
async def cta_autocomplete(
    current_user: CurrentUserOrDefault,
    q: Annotated[str, Query(max_length=512)] = "",
    model_id: str | None = None,
    tag_filter: CtaAutocompleteFilter | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[CtaAutocompleteCandidate]:
    """Search for CTA tags, optionally paging through a category filter."""
    service = _get_service()

    try:
        return await run_in_threadpool(service.autocomplete, q, model_id, tag_filter, offset, limit)
    except CtaUnavailableError:
        raise HTTPException(status_code=503, detail="CTA service not available")


@clip_tag_autocomplete_router.get(
    "/tags",
    operation_id="list_cta_tags",
    response_model=CtaTagPage,
)
async def list_cta_tags(
    current_user: AdminUserOrDefault,
    q: Annotated[str | None, Query(max_length=512)] = None,
    tag_type: CtaTagType | None = None,
    tag_set_id: str | None = None,
    uncategorized: bool | None = None,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> CtaTagPage:
    """List CTA tags with server-side filtering and cursor pagination. Admin only."""
    service = _get_service()

    try:
        return await run_in_threadpool(
            service.list_tags,
            q=q,
            tag_type=tag_type,
            tag_set_id=tag_set_id,
            uncategorized=uncategorized,
            cursor=cursor,
            limit=limit,
        )
    except (CtaUnavailableError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.get(
    "/tag_sets",
    operation_id="list_cta_tag_sets",
    response_model=list[CtaTagSetDTO],
)
async def list_cta_tag_sets(
    current_user: AdminUserOrDefault,
) -> list[CtaTagSetDTO]:
    """List CTA tag sets."""
    service = _get_service()
    try:
        return await run_in_threadpool(service.list_tag_sets)
    except CtaUnavailableError:
        raise HTTPException(status_code=503, detail="CTA service not available")


@clip_tag_autocomplete_router.get(
    "/syntax_profiles",
    operation_id="list_cta_syntax_profiles",
    response_model=list[CtaSyntaxProfileDTO],
)
async def list_cta_syntax_profiles(
    current_user: AdminUserOrDefault,
) -> list[CtaSyntaxProfileDTO]:
    """List CTA syntax profiles."""
    service = _get_service()
    try:
        return await run_in_threadpool(service.list_syntax_profiles)
    except CtaUnavailableError:
        raise HTTPException(status_code=503, detail="CTA service not available")


# ---------------------------------------------------------------------------
# Tag detail / update / delete
# ---------------------------------------------------------------------------


@clip_tag_autocomplete_router.get(
    "/tags/i/{tag_id}",
    operation_id="get_cta_tag",
    response_model=CtaTagDetailDTO,
)
async def get_cta_tag(
    current_user: AdminUserOrDefault,
    tag_id: str,
) -> CtaTagDetailDTO:
    service = _get_service()
    try:
        return await run_in_threadpool(service.get_tag, tag_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Tag not found")


@clip_tag_autocomplete_router.patch(
    "/tags/i/{tag_id}",
    operation_id="update_cta_tag",
    response_model=CtaTagMutationResult,
)
async def update_cta_tag(
    current_user: AdminUserOrDefault,
    tag_id: str,
    body: CtaTagUpdate,
) -> CtaTagMutationResult:
    service = _get_service()
    try:
        return await run_in_threadpool(
            service.update_tag,
            tag_id,
            canonical_content=body.canonical_content,
            tag_type=body.tag_type,
            popularity=body.popularity,
            popularity_supplied="popularity" in body.model_fields_set,
            tag_set_ids=body.tag_set_ids,
            tag_set_ids_supplied="tag_set_ids" in body.model_fields_set,
        )
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.delete(
    "/tags/i/{tag_id}",
    operation_id="delete_cta_tag",
)
async def delete_cta_tag(
    current_user: AdminUserOrDefault,
    tag_id: str,
) -> None:
    service = _get_service()
    try:
        await run_in_threadpool(service.delete_tag, tag_id)
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


# ---------------------------------------------------------------------------
# Bulk tag mutations
# ---------------------------------------------------------------------------


@clip_tag_autocomplete_router.post(
    "/tags/bulk",
    operation_id="bulk_mutate_cta_tags",
    response_model=CtaBulkResult,
)
async def bulk_mutate_cta_tags(
    current_user: AdminUserOrDefault,
    body: CtaBulkTagRequest,
) -> CtaBulkResult:
    service = _get_service()
    try:
        return await run_in_threadpool(service.bulk_mutate_tags, body)
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


# ---------------------------------------------------------------------------
# Tag Sets CRUD
# ---------------------------------------------------------------------------


@clip_tag_autocomplete_router.post(
    "/tag_sets",
    operation_id="create_cta_tag_set",
    response_model=CtaTagSetDTO,
)
async def create_cta_tag_set(
    current_user: AdminUserOrDefault,
    body: CtaTagSetCreate,
) -> CtaTagSetDTO:
    service = _get_service()
    try:
        return await run_in_threadpool(service.create_tag_set, body.name)
    except (CtaUnavailableError, CtaWriteBusyError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.patch(
    "/tag_sets/i/{tag_set_id}",
    operation_id="update_cta_tag_set",
    response_model=CtaTagSetDTO,
)
async def update_cta_tag_set(
    current_user: AdminUserOrDefault,
    tag_set_id: str,
    body: CtaTagSetUpdate,
) -> CtaTagSetDTO:
    service = _get_service()
    try:
        return await run_in_threadpool(service.update_tag_set, tag_set_id, body.name)
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.delete(
    "/tag_sets/i/{tag_set_id}",
    operation_id="delete_cta_tag_set",
)
async def delete_cta_tag_set(
    current_user: AdminUserOrDefault,
    tag_set_id: str,
) -> None:
    service = _get_service()
    try:
        await run_in_threadpool(service.delete_tag_set, tag_set_id)
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


# ---------------------------------------------------------------------------
# Syntax Profiles CRUD
# ---------------------------------------------------------------------------


@clip_tag_autocomplete_router.post(
    "/syntax_profiles",
    operation_id="create_cta_syntax_profile",
    response_model=CtaSyntaxProfileDTO,
)
async def create_cta_syntax_profile(
    current_user: AdminUserOrDefault,
    body: CtaSyntaxProfileCreate,
) -> CtaSyntaxProfileDTO:
    service = _get_service()
    try:
        return await run_in_threadpool(
            service.create_syntax_profile,
            **body.model_dump(),
        )
    except (CtaUnavailableError, CtaWriteBusyError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.patch(
    "/syntax_profiles/i/{profile_id}",
    operation_id="update_cta_syntax_profile",
    response_model=CtaSyntaxProfileDTO,
)
async def update_cta_syntax_profile(
    current_user: AdminUserOrDefault,
    profile_id: str,
    body: CtaSyntaxProfileUpdate,
) -> CtaSyntaxProfileDTO:
    service = _get_service()
    try:
        return await run_in_threadpool(
            service.update_syntax_profile,
            profile_id,
            **body.model_dump(exclude_unset=True),
        )
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.delete(
    "/syntax_profiles/i/{profile_id}",
    operation_id="delete_cta_syntax_profile",
)
async def delete_cta_syntax_profile(
    current_user: AdminUserOrDefault,
    profile_id: str,
) -> None:
    service = _get_service()
    try:
        await run_in_threadpool(service.delete_syntax_profile, profile_id)
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@clip_tag_autocomplete_router.get(
    "/models/i/{model_id}/config",
    operation_id="get_cta_model_config",
    response_model=CtaModelConfigDTO,
)
async def get_cta_model_config(
    current_user: AdminUserOrDefault,
    model_id: str,
) -> CtaModelConfigDTO:
    service = _get_service()
    if not ApiDependencies.invoker.services.model_manager.store.exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return await run_in_threadpool(service.get_model_config, model_id)


@clip_tag_autocomplete_router.put(
    "/models/i/{model_id}/config",
    operation_id="set_cta_model_config",
    response_model=CtaModelConfigDTO,
)
async def set_cta_model_config(
    current_user: AdminUserOrDefault,
    model_id: str,
    body: CtaModelConfigUpdate,
) -> CtaModelConfigDTO:
    service = _get_service()
    if not ApiDependencies.invoker.services.model_manager.store.exists(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        return await run_in_threadpool(
            service.set_model_config,
            model_id,
            syntax_profile_id=body.syntax_profile_id,
            tag_set_ids=body.tag_set_ids,
        )
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


# ---------------------------------------------------------------------------
# Sample CSV
# ---------------------------------------------------------------------------


@clip_tag_autocomplete_router.get(
    "/imports/sample",
    operation_id="get_cta_sample_csv",
)
async def get_cta_sample_csv(current_user: AdminUserOrDefault) -> Response:
    sample_csv = """tag,popularity,type
example_character,25000,character
example_artist,5000,artist
red_hair,120000,general
"""
    return Response(
        content=sample_csv.encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cta_sample_tags.csv"},
    )


# ---------------------------------------------------------------------------
# Import endpoints
# ---------------------------------------------------------------------------


@clip_tag_autocomplete_router.post(
    "/imports/stage",
    operation_id="stage_cta_import",
    response_model=CtaImportStageDTO,
)
async def stage_cta_import(
    current_user: AdminUserOrDefault,
    file: Annotated[UploadFile, File()],
) -> CtaImportStageDTO:
    """Stage a CSV file for import. Returns detection results and sample."""
    service = _get_service()

    try:
        result = await run_in_threadpool(
            service.stage_import,
            current_user.user_id,
            file.file,
        )
        return result
    except (CtaUnavailableError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.post(
    "/imports/i/{session_id}/prepare",
    operation_id="prepare_cta_import",
    response_model=CtaImportPreviewDTO,
)
async def prepare_cta_import(
    current_user: AdminUserOrDefault,
    session_id: str,
    body: CtaImportMapping,
) -> CtaImportPreviewDTO:
    """Prepare (full parse) a staged import."""
    service = _get_service()

    try:
        return await run_in_threadpool(
            service.prepare_import,
            current_user.user_id,
            session_id,
            tag_column=body.tag_column,
            popularity_column=body.popularity_column,
            type_column=body.type_column,
            delimiter=body.delimiter,
            first_row_contains_column_names=body.first_row_contains_column_names,
        )
    except (CtaUnavailableError, LookupError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.post(
    "/imports/i/{session_id}/commit",
    operation_id="commit_cta_import",
    response_model=CtaImportResultDTO,
)
async def commit_cta_import(
    current_user: AdminUserOrDefault,
    session_id: str,
    body: CtaImportCommitRequest,
) -> CtaImportResultDTO:
    """Commit a prepared import to the CTA database."""
    service = _get_service()

    try:
        return await run_in_threadpool(
            service.commit_import,
            current_user.user_id,
            session_id,
            destination=body.destination.model_dump(),
        )
    except (CtaUnavailableError, CtaWriteBusyError, LookupError, ValueError) as e:
        _raise_cta_error(e)


@clip_tag_autocomplete_router.delete(
    "/imports/i/{session_id}",
    operation_id="cancel_cta_import",
)
async def cancel_cta_import(
    current_user: AdminUserOrDefault,
    session_id: str,
) -> None:
    """Cancel and clean up an import session."""
    service = _get_service()

    try:
        await run_in_threadpool(service.cancel_import, current_user.user_id, session_id)
    except (CtaUnavailableError, LookupError, ValueError) as e:
        _raise_cta_error(e)
