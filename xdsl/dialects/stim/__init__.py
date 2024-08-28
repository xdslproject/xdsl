from xdsl.ir import Dialect

from .ops import StimCircuitOp

Stim = Dialect(
    "stim",
    # first list operations to include in the dialect
    [
        StimCircuitOp,
    ],
)
