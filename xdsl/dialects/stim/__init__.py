from xdsl.ir import Dialect

from .ops import QubitAttr, QubitMappingAttr, StimCircuitOp

Stim = Dialect(
    "stim",
    [
        StimCircuitOp,
        QubitCoordsOp,
    ],
    [
        QubitAttr,
        QubitMappingAttr,
    ],
)
