from xdsl.ir import Dialect

from .ops import (
    CliffordGateOp,
    MeasurementGateOp,
    PauliAttr,
    QubitCoordsOp,
    QubitMappingAttr,
    ResetGateOp,
    SingleQubitGateAttr,
    StimCircuitOp,
    TwoQubitGateAttr,
)

Stim = Dialect(
    "stim",
    [
        CliffordGateOp,
        MeasurementGateOp,
        QubitCoordsOp,
        ResetGateOp,
        StimCircuitOp,
    ],
    [PauliAttr, QubitMappingAttr, SingleQubitGateAttr, TwoQubitGateAttr],
)
