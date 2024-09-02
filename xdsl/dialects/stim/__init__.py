from xdsl.ir import Dialect

from .ops import StimCircuitOp

Stim = Dialect(
    "stim",
    [
        StimCircuitOp,
    ],
)
