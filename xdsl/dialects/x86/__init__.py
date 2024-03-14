from xdsl.ir import Dialect

from .ops import *
from .register import *

X86 = Dialect(
    "x86",
    [],
    [
        GeneralRegisterType,
    ],
)
