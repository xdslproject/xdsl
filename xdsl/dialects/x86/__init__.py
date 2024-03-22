from xdsl.ir import Dialect

from .register import GeneralRegisterType

X86 = Dialect(
    "x86",
    [],
    [
        GeneralRegisterType,
    ],
)
