from xdsl.ir import Dialect

from .register import *

X86 = Dialect(
    "x86",
    [],
    [
        GeneralRegisterType,  # noqa: F405
    ],
)
