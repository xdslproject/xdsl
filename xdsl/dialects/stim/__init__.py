from xdsl.ir import Dialect

from .ops import (
    CliffordGateOp,
    MeasurementGateOp,
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
        MeasurementGateOp,
    ],
    [PauliAttr, QubitMappingAttr, SingleQubitGateAttr, TwoQubitGateAttr],
)
