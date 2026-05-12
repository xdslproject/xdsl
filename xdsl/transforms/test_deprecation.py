import warnings

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass


class TestDeprecationPass(ModulePass):
    """
    Test pass that does nothing and raises a DeprecationWarning.
    """

    name = "test-deprecation"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        warnings.warn("hello", DeprecationWarning)
