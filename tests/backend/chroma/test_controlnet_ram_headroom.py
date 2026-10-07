from unittest.mock import MagicMock

import pytest

from invokeai.app.services.shared.invocation_context import ModelsInterface


def test_models_interface_make_room_in_ram_cache_delegates_to_model_cache() -> None:
    services = MagicMock()
    interface = ModelsInterface(services=services, data=MagicMock(), util=MagicMock())

    interface.make_room_in_ram_cache(4 * 2**30)

    services.model_manager.load.ram_cache.make_room.assert_called_once_with(4 * 2**30)


def test_models_interface_make_room_in_ram_cache_rejects_negative_values() -> None:
    services = MagicMock()
    interface = ModelsInterface(services=services, data=MagicMock(), util=MagicMock())

    with pytest.raises(ValueError, match="bytes_needed must be non-negative"):
        interface.make_room_in_ram_cache(-1)

    services.model_manager.load.ram_cache.make_room.assert_not_called()
