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
from xdsl.dialects.riscv.abstract_ops import (
    GetAnyRegisterOperation,
    LdOperation,
    LiOperation,
    SdOperation,
)
from xdsl.dialects.riscv.attrs import I12, i12
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    SSAValue,
)
from xdsl.irdl import (
    irdl_op_definition,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.traits import MemoryReadEffect


@irdl_op_definition
class LiOp(LiOperation[I64]):
    """
    Loads a 64-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc).
    """

    name = "rv64.li"

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
class LdOp(LdOperation[I12]):
    """
    Loads a 64-bit value from memory into register rd for RV64I.
    ```C
    x[rd] = M[x[rs1] + sext(offset)][63:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/#_ld).
    """

    name = "rv64.ld"

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[I12],
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i12)
        super().__init__(rs1, immediate, rd=rd, comment=comment)


@irdl_op_definition
class SdOp(SdOperation[I12]):
    """
    Store 64-bit, values from register rs2 to memory.
    ```C
    M[x[rs1] + sext(offset)] = x[rs2][63:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/#_sd).
    """

    name = "rv64.sd"

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        immediate: int | IntegerAttr[I12],
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i12)
        super().__init__(rs1, rs2, immediate, comment=comment)


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[IntRegisterType]):
    name = "rv64.get_register"


RV64 = Dialect(
    "rv64",
    [
        LiOp,
        LdOp,
        SdOp,
        GetRegisterOp,
    ],
    [],
)
