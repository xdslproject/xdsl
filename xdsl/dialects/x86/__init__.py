from xdsl.ir import Dialect

from .ops import AddOp, GetRegisterOp
from .register import GeneralRegisterType

X86 = Dialect(
    "x86",
    [AddOp, GetRegisterOp],
    [
        GeneralRegisterType,
    ],
)
