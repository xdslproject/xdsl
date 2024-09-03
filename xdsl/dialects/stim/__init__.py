from xdsl.ir import Dialect

from .ops import (
    CliffordGateOp,
    PauliAttr,
    QubitCoordsOp,
    QubitMappingAttr,
    SingleQubitGateAttr,
    StimCircuitOp,
    TwoQubitGateAttr,
)

Stim = Dialect(
    "stim",
    [
        CliffordGateOp,
        QubitCoordsOp,
        StimCircuitOp,
    ],
    [PauliAttr, QubitMappingAttr, SingleQubitGateAttr, TwoQubitGateAttr],
)
