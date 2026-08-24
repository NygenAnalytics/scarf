"""Optional plotting dependency error tests."""

import builtins

import pytest

from scarf.plotting._deps import (
    require_kneed,
    require_matplotlib,
    require_seaborn,
)


@pytest.mark.parametrize(
    ("loader", "blocked_package", "message"),
    [
        (require_matplotlib, "matplotlib", "matplotlib"),
        (require_seaborn, "seaborn", "seaborn"),
        (require_kneed, "kneed", "kneed"),
    ],
)
def test_optional_dependency_errors_are_actionable(
    monkeypatch,
    loader,
    blocked_package,
    message,
):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".", 1)[0] == blocked_package:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=message):
        loader()
