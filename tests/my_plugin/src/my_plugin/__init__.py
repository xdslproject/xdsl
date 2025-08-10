from xdsl.ir import Dialect
from xdsl.passes import ModulePass
from xdsl.universe import Universe


class PluginPass(ModulePass):
    name = "plugin-pass"


PluginDialect = Dialect("plugin_dialect")

MY_UNIVERSE = Universe(
    {"plugin_dialect": lambda: PluginDialect},
    {"plugin-pass": lambda: PluginPass},
)
