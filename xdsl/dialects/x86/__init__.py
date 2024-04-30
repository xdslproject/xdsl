from xdsl.ir import Dialect

from .ops import (
    GetRegisterOp,
    R_NotOp,
    R_PopOp,
    R_PushOp,
    RM_AddOp,
    RM_AndOp,
    RM_ImulOp,
    RM_MovOp,
    RM_OrOp,
    RM_SubOp,
    RM_XorOp,
    RR_AddOp,
    RR_AndOp,
    RR_ImulOp,
    RR_MovOp,
    RR_OrOp,
    RR_SubOp,
    RR_XorOp,
)
from .register import GeneralRegisterType

X86 = Dialect(
    "x86",
    [
        RR_AddOp,
        RR_SubOp,
        RR_ImulOp,
        RR_AndOp,
        RR_OrOp,
        RR_XorOp,
        RR_MovOp,
        R_PushOp,
        R_PopOp,
        R_NotOp,
        RM_AddOp,
        RM_SubOp,
        RM_ImulOp,
        RM_AndOp,
        RM_OrOp,
        RM_XorOp,
        RM_MovOp,
        GetRegisterOp,
    ],
    [
        GeneralRegisterType,
    ],
)
