import re

import pytest

from xdsl.ir import Dialect
from xdsl.universe import XDSL_UNIVERSE, Universe


def test_multiverse():
    """
    Checks that the universe can be discovered within the current environment, and that
    duplicate names are detected.
    """
    multiverse = Universe.get_multiverse()

    assert "plugin_dialect" in multiverse.all_dialects
    assert "plugin-pass" in multiverse.all_passes

    import my_plugin

    XDSL_UNIVERSE.all_dialects["plugin_dialect"] = lambda: Dialect("plugin_dialect")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Duplicate definition of plugin_dialect in ['my_plugin', 'xdsl']."
        ),
    ):
        Universe.get_multiverse()

    del XDSL_UNIVERSE.all_dialects["plugin_dialect"]

    XDSL_UNIVERSE.all_passes["plugin-pass"] = lambda: my_plugin.PluginPass

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Duplicate definition of plugin-pass in ['my_plugin', 'xdsl']."
        ),
    ):
        Universe.get_multiverse()

    del XDSL_UNIVERSE.all_passes["plugin-pass"]
