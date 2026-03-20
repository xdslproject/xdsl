"""
RISC-V 64-bit (RV64) dialect operations and types.

This module defines the RV64-specific variant of RISC-V operations,
using 6-bit immediates for 64-bit architectures.
"""

from __future__ import annotations

from xdsl.dialects.builtin import I64, IntegerAttr, StringAttr, i64
from xdsl.dialects.riscv import (
    IntRegisterType,
    LabelAttr,
    Registers,
    parse_immediate_value,
)
from xdsl.dialects.riscv.abstract_ops import GetAnyRegisterOperation, LiOperation
from xdsl.dialects.riscv.ops import LiOpHasCanonicalizationPatternTrait
from xdsl.ir import (
    Attribute,
    Dialect,
)
from xdsl.irdl import (
    irdl_op_definition,
    traits_def,
)
from xdsl.parser import Parser


@irdl_op_definition
class LiOp(LiOperation[I64]):
    """
    Loads a 64-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc).
    """

    name = "rv64.li"

    traits = traits_def(LiOpHasCanonicalizationPatternTrait())

    def __init__(
        self,
        immediate: int | IntegerAttr[I64] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i64)
        super().__init__(immediate, rd=rd, comment=comment)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i64)
        return attributes


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[IntRegisterType]):
    name = "rv64.get_register"


RV64 = Dialect(
    "rv64",
    [
        LiOp,
        GetRegisterOp,
    ],
    [],
)
