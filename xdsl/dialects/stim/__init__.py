from xdsl.ir import Dialect

from .ops import (
    HOp,
    IOp,
    QubitAttr,
    QubitCoordsOp,
    QubitMappingAttr,
    SDagOp,
    SOp,
    SqrtXDagOp,
    SqrtXOp,
    SqrtYDagOp,
    SqrtYOp,
    StimCircuitOp,
    XOp,
    YOp,
    ZOp,
)

Stim = Dialect(
    "stim",
    [
        HOp,
        IOp,
        QubitCoordsOp,
        SDagOp,
        SOp,
        SqrtXDagOp,
        SqrtXOp,
        SqrtYDagOp,
        SqrtYOp,
        StimCircuitOp,
        XOp,
        YOp,
        ZOp,
    ],
    [
        QubitAttr,
        QubitMappingAttr,
    ],
)
