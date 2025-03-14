from __future__ import annotations

from abc import ABC

from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.dialects.arm.assembly import assembly_arg_str
from xdsl.dialects.arm.ops import ARMInstruction, ARMOperation
from xdsl.dialects.arm.register import ARMRegisterType
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)

ARM_NEON_INDEX_BY_NAME = {f"v{i}": i for i in range(0, 32)}


@irdl_attr_definition
class NEONRegisterType(ARMRegisterType):
    """
    A 128-bit NEON ARM register type.
    """

    name = "arm_neon.reg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "arm_neon"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return ARM_NEON_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_"


UNALLOCATED_NEON = NEONRegisterType.unallocated()
V0 = NEONRegisterType.from_name("v0")
V1 = NEONRegisterType.from_name("v1")
V2 = NEONRegisterType.from_name("v2")
V3 = NEONRegisterType.from_name("v3")
V4 = NEONRegisterType.from_name("v4")
V5 = NEONRegisterType.from_name("v5")
V6 = NEONRegisterType.from_name("v6")
V7 = NEONRegisterType.from_name("v7")
V8 = NEONRegisterType.from_name("v8")
V9 = NEONRegisterType.from_name("v9")
V10 = NEONRegisterType.from_name("v10")
V11 = NEONRegisterType.from_name("v11")
V12 = NEONRegisterType.from_name("v12")
V13 = NEONRegisterType.from_name("v13")
V14 = NEONRegisterType.from_name("v14")
V15 = NEONRegisterType.from_name("v15")
V16 = NEONRegisterType.from_name("v16")
V17 = NEONRegisterType.from_name("v17")
V18 = NEONRegisterType.from_name("v18")
V19 = NEONRegisterType.from_name("v19")
V20 = NEONRegisterType.from_name("v20")
V21 = NEONRegisterType.from_name("v21")
V22 = NEONRegisterType.from_name("v22")
V23 = NEONRegisterType.from_name("v23")
V24 = NEONRegisterType.from_name("v24")
V25 = NEONRegisterType.from_name("v25")
V26 = NEONRegisterType.from_name("v26")
V27 = NEONRegisterType.from_name("v27")
V28 = NEONRegisterType.from_name("v28")
V29 = NEONRegisterType.from_name("v29")
V30 = NEONRegisterType.from_name("v30")
V31 = NEONRegisterType.from_name("v31")


class ARMNEONInstruction(ARMInstruction, ABC):
    """
    Base class for operations in the NEON instruction set.
    The name of the operation will be used as the NEON assembly instruction name.

    The arrangement specifier for NEON instructions determines element size and count:
      - "4H"  → 4 half-precision floats
      - "8H"  → 8 half-precision floats
      - "2S"  → 2 single-precision floats
      - "4S"  → 4 single-precision floats
      - "2D"  → 2 double-precision floats
    """

    arrangement = attr_def(StringAttr)

    def assembly_line(self) -> str | None:
        # default assembly code generator
        instruction_name = self.assembly_instruction_name()
        arg_str = ", ".join(
            f"{assembly_arg_str(arg)}.{self.arrangement.data}"
            for arg in self.assembly_line_args()
            if arg is not None
        )
        return AssemblyPrinter.assembly_line(instruction_name, arg_str, self.comment)


@irdl_op_definition
class GetRegisterOp(ARMOperation):
    """
    This instruction allows us to create an SSAValue for a given register name.
    """

    name = "arm_neon.get_register"

    result = result_def(NEONRegisterType)
    assembly_format = "attr-dict `:` type($result)"

    def __init__(self, register_type: NEONRegisterType):
        super().__init__(result_types=[register_type])

    def assembly_line(self):
        return None


@irdl_op_definition
class DSSFMulVecOp(ARMNEONInstruction):
    """
    Floating-point multiply (vector)
    This instruction multiplies corresponding floating-point values in the vectors in the two source SIMD&FP
    registers, places the result in a vector, and writes the vector to the destination SIMD&FP register.
    Encoding: FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<T>.
    Vd, Vn, Vm specify the SIMD&FP regs. The <T> specifier determines element arrangement (size and count).
    https://developer.arm.com/documentation/ddi0602/2024-12/SIMD-FP-Instructions/FMUL--vector---Floating-point-multiply--vector--?lang=en#T_option__4
    """

    name = "arm_neon.dss.fmulvec"
    d = result_def(NEONRegisterType)
    s1 = operand_def(NEONRegisterType)
    s2 = operand_def(NEONRegisterType)

    assembly_format = (
        "$s1 `,` $s2 attr-dict `:` `(` type($s1) `,` type($s2) `)` `->` type($d)"
    )

    def __init__(
        self,
        s1: Operation | SSAValue,
        s2: Operation | SSAValue,
        *,
        d: NEONRegisterType,
        arrangement: str | StringAttr,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(arrangement, str):
            valid_arrangements = {"4H", "8H", "2S", "4S", "2D"}
            if arrangement in valid_arrangements:
                arrangement = StringAttr(arrangement)
            else:
                raise ValueError(f"Invalid FMUL arrangement: {arrangement}")
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=(s1, s2),
            attributes={
                "arrangement": arrangement,
                "comment": comment,
            },
            result_types=(d,),
        )

    def assembly_instruction_name(self) -> str:
        return "fmul"

    def assembly_line_args(self):
        return (self.d, self.s1, self.s2)

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        arg_str = ", ".join(
            f"{assembly_arg_str(arg)}.{self.arrangement.data}"
            for arg in self.assembly_line_args()
            if arg is not None
        )
        return AssemblyPrinter.assembly_line(instruction_name, arg_str, self.comment)


ARM_NEON = Dialect(
    "arm_neon",
    [
        DSSFMulVecOp,
        GetRegisterOp,
    ],
    [
        NEONRegisterType,
    ],
)
