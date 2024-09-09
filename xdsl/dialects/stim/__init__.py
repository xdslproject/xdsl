from xdsl.ir import Dialect

from .ops import (
    CliffordGateOp,
    DepolarizingNoiseAttr,
    MeasurementGateOp,
    ObservableIncludeOp,
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
        ObservableIncludeOp,
        QubitCoordsOp,
        ResetGateOp,
        StimCircuitOp,
        TickAnnotationOp,
    ],
    [
        PauliAttr,
        QubitMappingAttr,
        SingleQubitGateAttr,
        TwoQubitGateAttr,
        DepolarizingNoiseAttr,
    ],
)
