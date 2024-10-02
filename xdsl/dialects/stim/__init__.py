from xdsl.ir import Dialect

from .ops import QubitAttr, QubitCoordsOp, QubitMappingAttr, StimCircuitOp

Stim = Dialect(
    "stim",
    [
        QubitCoordsOp,
        StimCircuitOp,
    ],
    [
        QubitAttr,
        QubitMappingAttr,
    ],
)
