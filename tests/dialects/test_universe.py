import re

import pytest
from my_plugin import Dialect

from xdsl.universe import Universe


def test_multiverse():
    multiverse = Universe.get_multiverse()

    assert "plugin_dialect" in multiverse.all_dialects
    assert "plugin-pass" in multiverse.all_passes

    import my_plugin

    my_plugin.MY_UNIVERSE.all_dialects["scf"] = lambda: Dialect("scf")

    with pytest.raises(
        ValueError,
        match=re.escape("Duplicate definition of scf in ['my_plugin', 'xdsl']."),
    ):
        Universe.get_multiverse()

    del my_plugin.MY_UNIVERSE.all_dialects["scf"]

    my_plugin.MY_UNIVERSE.all_passes["dce"] = lambda: my_plugin.PluginPass

    with pytest.raises(
        ValueError,
        match=re.escape("Duplicate definition of dce in ['my_plugin', 'xdsl']."),
    ):
        Universe.get_multiverse()

    del my_plugin.MY_UNIVERSE.all_passes["dce"]
