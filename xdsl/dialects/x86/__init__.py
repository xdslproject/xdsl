from xdsl.ir import Dialect

from .ops import AddOp, AndOp, GetRegisterOp, ImulOp, MovOp, OrOp, SubOp, XorOp
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
        GetRegisterOp,
    ],
    [
        GeneralRegisterType,
    ],
)
