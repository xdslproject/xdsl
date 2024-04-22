from xdsl.ir import Dialect

from .ops import (
    AddOp,
    AndOp,
    GetRegisterOp,
    ImulOp,
    MovOp,
    NotOp,
    OrOp,
    PopOp,
    PushOp,
    SubOp,
    XorOp,
)
from .register import GeneralRegisterType

X86 = Dialect(
    "x86",
    [
        AddOp,
        SubOp,
        ImulOp,
        AndOp,
        OrOp,
        XorOp,
        MovOp,
        PushOp,
        PopOp,
        NotOp,
        GetRegisterOp,
    ],
    [
        GeneralRegisterType,
    ],
)
