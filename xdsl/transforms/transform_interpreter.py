"""Transform interpreter."""

from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.passes import Context, ModulePass


@dataclass(frozen=True)
class TransformInterpreterPass(ModulePass):
    """Transform dialect interpreter"""

    name = "transform-interpreter"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None: ...
