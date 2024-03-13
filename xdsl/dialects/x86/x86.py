from __future__ import annotations

from xdsl.ir import Dialect

from .register import GeneralRegisterType, Registers

# This is where all of the instructions will be placed.
Registers = Registers  # here to allow the test file to import the Registers class
# let me know if you have better ideas on how to do this

X86 = Dialect(
    "x86",
    [],
    [
        GeneralRegisterType,
    ],
)
