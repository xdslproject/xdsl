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
    TickAnnotationOp,
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
        TickAnnotationOp,
    ],
    [PauliAttr, QubitMappingAttr, SingleQubitGateAttr, TwoQubitGateAttr],
)
