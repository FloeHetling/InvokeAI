"""Who may use the CLIP tag autocomplete routes.

Everyone signed in may search tags and read the status; importing, editing and deleting tag data, and
choosing a model's tag sets and syntax profile, is administration. The routes check this through a dependency
that runs before any handler code, so these tests need no CTA service: a non-admin must be refused
before the handler would even look for one.
"""

from collections.abc import Iterator

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from invokeai.app.api.auth_dependencies import get_current_user_or_default
from invokeai.app.api.routers.clip_tag_autocomplete import clip_tag_autocomplete_router
from invokeai.app.api_app import app
from invokeai.app.services.auth.token_service import TokenData

# What any signed-in user may call.
USER_OPERATION_IDS = {"get_cta_status", "cta_autocomplete"}

PATH_PARAMETER_VALUES = {
    "tag_id": "tag-1",
    "tag_set_id": "set-1",
    "profile_id": "profile-1",
    "model_id": "model-1",
    "session_id": "session-1",
}


def _management_routes() -> list[tuple[str, str]]:
    routes: list[tuple[str, str]] = []

    for route in clip_tag_autocomplete_router.routes:
        assert isinstance(route, APIRoute)

        if route.operation_id in USER_OPERATION_IDS:
            continue

        path = "/api" + route.path.format(**PATH_PARAMETER_VALUES)

        routes.extend((method, path) for method in sorted(route.methods))

    return routes


@pytest.fixture
def non_admin_client() -> Iterator[TestClient]:
    user = TokenData(user_id="user-1", email="user@example.com", is_admin=False)
    app.dependency_overrides[get_current_user_or_default] = lambda: user

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user_or_default, None)


def test_every_route_but_search_and_status_is_listed_for_the_check() -> None:
    operation_ids = {route.operation_id for route in clip_tag_autocomplete_router.routes if isinstance(route, APIRoute)}

    assert USER_OPERATION_IDS <= operation_ids
    # Fails when a route is added: decide whether it is administration, then keep this list honest.
    assert len(_management_routes()) == len(operation_ids) - len(USER_OPERATION_IDS)


@pytest.mark.parametrize(("method", "path"), _management_routes())
def test_a_signed_in_user_who_is_not_an_admin_cannot_manage_tag_data(
    non_admin_client: TestClient, method: str, path: str
) -> None:
    response = non_admin_client.request(method, path, json={})

    assert response.status_code == 403
