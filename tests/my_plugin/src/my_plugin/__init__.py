from dataclasses import dataclass
from typing import IO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Dialect
from xdsl.passes import ModulePass
from xdsl.universe import Universe
from xdsl.utils.target import Target


class PluginPass(ModulePass):
    name = "plugin-pass"


@dataclass(frozen=True)
class PluginTarget(Target):
    name = "plugin-target"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        pass


PluginDialect = Dialect("plugin_dialect")

MY_UNIVERSE = Universe(
    all_dialects={"plugin_dialect": lambda: PluginDialect},
    all_passes={"plugin-pass": lambda: PluginPass},
    all_targets={"plugin-target": lambda: PluginTarget},
)
