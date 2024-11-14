from xdsl.ir import Dialect

from .attributes import LabelAttr
from .ops import GetRegisterOp, RR_MovOp
from .register import GeneralRegisterType

ARM = Dialect(
    "arm",
    [GetRegisterOp, RR_MovOp],
    [GeneralRegisterType, LabelAttr],
)
