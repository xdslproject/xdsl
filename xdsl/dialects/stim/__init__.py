from xdsl.ir import Dialect

from .ops import QubitCoordsOp, QubitMappingAttr, StimCircuitOp

Stim = Dialect(
    "stim",
    [
        QubitCoordsOp,
        StimCircuitOp,
    ],
    [
        QubitMappingAttr,
    ],
)
