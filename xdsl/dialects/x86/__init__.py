from xdsl.ir import Dialect

from .register import *  # noqa: F405

X86 = Dialect(
    "x86",
    [],
    [
        GeneralRegisterType,
    ],
)
