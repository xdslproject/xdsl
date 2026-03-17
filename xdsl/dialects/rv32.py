"""
RISC-V 32-bit (RV32) dialect operations and types.

This module defines the RV32-specific variant of RISC-V operations,
using 5-bit immediates for 32-bit architectures.
"""

from __future__ import annotations

from xdsl.dialects.builtin import I32, IntegerAttr, StringAttr, i32
from xdsl.dialects.riscv import (
    IntRegisterType,
    LabelAttr,
    Registers,
    parse_immediate_value,
)
from xdsl.dialects.riscv.abstract_ops import GetAnyRegisterOperation, LiOperation
from xdsl.ir import (
    Attribute,
    Dialect,
)
from xdsl.irdl import (
    irdl_op_definition,
)
from xdsl.parser import Parser


@irdl_op_definition
class LiOp(LiOperation[I32]):
    """
    Loads a 32-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc).
    """

    name = "rv32.li"

    def __init__(
        self,
        immediate: int | IntegerAttr[I32] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i32)
        super().__init__(immediate, rd=rd, comment=comment)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i32)
        return attributes


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[IntRegisterType]):
    name = "rv32.get_register"


RV32 = Dialect(
    "rv32",
    [
        LiOp,
        GetRegisterOp,
    ],
    [],
)
