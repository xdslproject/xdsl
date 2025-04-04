from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.arm.assembly import AssemblyInstructionArg, square_brackets_reg
from xdsl.dialects.arm.ops import ARMInstruction, ARMOperation
from xdsl.dialects.arm.register import ARMRegisterType, IntRegisterType
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i8
from xdsl.ir import (
    Dialect,
    EnumAttribute,
    Operation,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
)
from xdsl.irdl import (
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
)
from xdsl.utils.exceptions import VerifyException

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


class NeonArrangement(StrEnum):
    """
    The arrangement specifier for NEON instructions determines element size and count.
    We assume full 128-bit registers. Possible arrangements:
      - D  → 2 double-precision floats
      - S  → 4 single-precision floats
      - H  → 8 half-precision floats
    """

    D = "D"
    S = "S"
    H = "H"

    def map_to_num_els(self):
        map = {"D": 2, "S": 4, "H": 8}
        return map[self.name]


@irdl_attr_definition
class NeonArrangementAttr(EnumAttribute[NeonArrangement], SpacedOpaqueSyntaxAttribute):
    """
    Attribute containing the arrangement specification.
    """

    name = "arm_neon.arrangement"


class VectorWithArrangement(AssemblyInstructionArg):
    reg: NEONRegisterType
    arrangement: NeonArrangementAttr
    index: int | None = None

    def __init__(
        self,
        reg: NEONRegisterType | SSAValue,
        arrangement: NeonArrangementAttr,
        *,
        index: int | None = None,
    ):
        if isinstance(reg, SSAValue):
            assert isinstance(reg.type, NEONRegisterType)
            reg = reg.type

        self.reg = reg
        self.arrangement = arrangement
        self.index = index

    def assembly_str(self):
        if self.index is None:
            return f"{self.reg.register_name.data}.{self.arrangement.data.map_to_num_els()}{self.arrangement.data.name}"
        else:
            return f"{self.reg.register_name.data}.{self.arrangement.data.name}[{self.index}]"


class VariadicNeonRegArg(AssemblyInstructionArg):
    regs: Sequence[VectorWithArrangement]
    arrangement: NeonArrangementAttr

    def __init__(
        self,
        regs: Sequence[SSAValue],
        arrangement: NeonArrangementAttr,
    ):
        self.arrangement = arrangement
        vectors: Sequence[VectorWithArrangement] = []
        for reg in regs:
            assert isinstance(reg.type, NEONRegisterType)
            vectors.append(VectorWithArrangement(reg, self.arrangement))

        self.regs = vectors

    def assembly_str(self):
        return "{" + ", ".join(s.assembly_str() for s in self.regs) + "}"


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
class DSSFMulVecScalarOp(ARMInstruction):
    """
    Floating-point multiply (mixed: first source operand is a vector, second is a scalar. Destination is a vector)
    This instruction multiplies each of the floating-point values in the first source operand by the
    second source operand and writes the resulting values to the corresponding lanes of the destination.
    Encoding: FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<idx>.
    Vd, Vn, Vm specify the regs. The <T> specifier determines element arrangement (size and count).
    The <idx> specifier determines the index of Vm at which the second source operand (scalar) can be found,
    preceded by a size specifier.

    See external [documentation](https://developer.arm.com/documentation/ddi0602/2024-12/SIMD-FP-Instructions/FMUL--vector---Floating-point-multiply--vector--?lang=en#T_option__4).
    """

    name = "arm_neon.dss.fmulvec"
    d = result_def(NEONRegisterType)
    s1 = operand_def(NEONRegisterType)
    s2 = operand_def(NEONRegisterType)
    scalar_idx = attr_def(IntegerAttr[i8])
    arrangement = attr_def(NeonArrangementAttr)

    assembly_format = "$s1 `,` $s2 `[` $scalar_idx `]` $arrangement attr-dict `:` `(` type($s1) `,` type($s2) `)` `->` type($d)"

    def __init__(
        self,
        s1: Operation | SSAValue,
        s2: Operation | SSAValue,
        *,
        d: NEONRegisterType,
        arrangement: NeonArrangement | NeonArrangementAttr,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(arrangement, NeonArrangement):
            arrangement = NeonArrangementAttr(arrangement)
        super().__init__(
            operands=(s1, s2),
            attributes={
                "comment": comment,
                "arrangement": arrangement,
            },
            result_types=(d,),
        )

    def assembly_instruction_name(self) -> str:
        return "fmul"

    def assembly_line_args(self):
        return (
            VectorWithArrangement(self.d, self.arrangement),
            VectorWithArrangement(self.s1, self.arrangement),
            VectorWithArrangement(
                self.s2, self.arrangement, index=self.scalar_idx.value.data
            ),
        )


@irdl_op_definition
class DSSFmlaVecScalarOp(ARMInstruction):
    """
    Floating-point fused Multiply-Add to accumulator (mixed: first source operand is a vector, second is a scalar.
    Destination is a vector)
    This instruction multiplies the values in the first source operand by the second source operand,
    adds the accumulated value from the destination operand, and writes the resulting values to the destination.
    Encoding: FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<idx>.
    Vd, Vn, Vm specify the regs. The <T> specifier determines element arrangement (size and count).
    The <idx> specifier determines the index of Vm at which the second source operand (scalar) can be found,
    preceded by a size specifier.

    See external [documentation](https://developer.arm.com/documentation/100069/0606/SIMD-Vector-Instructions/FMLA--vector-).
    """

    name = "arm_neon.dss.fmla"
    d = result_def(NEONRegisterType)
    s1 = operand_def(NEONRegisterType)
    s2 = operand_def(NEONRegisterType)
    scalar_idx = attr_def(IntegerAttr[i8])
    arrangement = attr_def(NeonArrangementAttr)

    assembly_format = "$s1 `,` $s2 `[` $scalar_idx `]` $arrangement attr-dict `:` `(` type($s1) `,` type($s2) `)` `->` type($d)"

    def __init__(
        self,
        s1: Operation | SSAValue,
        s2: Operation | SSAValue,
        *,
        d: NEONRegisterType,
        arrangement: NeonArrangement | NeonArrangementAttr,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(arrangement, NeonArrangement):
            arrangement = NeonArrangementAttr(arrangement)
        super().__init__(
            operands=(s1, s2),
            attributes={
                "comment": comment,
                "arrangement": arrangement,
            },
            result_types=(d,),
        )

    def assembly_instruction_name(self) -> str:
        return "fmla"

    def assembly_line_args(self):
        return (
            VectorWithArrangement(self.d, self.arrangement),
            VectorWithArrangement(self.s1, self.arrangement),
            VectorWithArrangement(
                self.s2, self.arrangement, index=self.scalar_idx.value.data
            ),
        )


@irdl_op_definition
class DVarSSt1Op(ARMInstruction):
    """
    Neon structure store instruction reads data from memory into 64-bit Neon registers.
    ST1 stores one to four registers of data to memory, with no interleaving.
    """

    name = "arm_neon.dvars.st1"
    d = operand_def(IntRegisterType)
    src_regs = var_operand_def(NEONRegisterType)
    arrangement = attr_def(NeonArrangementAttr)

    assembly_format = "$src_regs ` ` `[` $d `]` $arrangement attr-dict `:` `(` type($src_regs) `)` `->` type($d)"

    def __init__(
        self,
        d: IntRegisterType,
        src_regs: Sequence[SSAValue],
        *,
        arrangement: NeonArrangement | NeonArrangementAttr,
        comment: str | StringAttr | None = None,
    ):
        if not (1 <= len(self.src_regs) <= 4):
            raise ValueError(
                f"src_regs must contain between 1 and 4 elements, but got {len(self.src_regs)}."
            )
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(arrangement, NeonArrangement):
            arrangement = NeonArrangementAttr(arrangement)
        super().__init__(
            operands=[*src_regs],
            attributes={
                "comment": comment,
                "arrangement": arrangement,
            },
            result_types=(d,),
        )

    def verify_(self) -> None:
        if not (1 <= len(self.src_regs) <= 4):
            raise VerifyException(
                f"src_regs must contain between 1 and 4 elements, but got {len(self.src_regs)}."
            )

    def assembly_instruction_name(self) -> str:
        return "st1"

    def assembly_line_args(self):
        assert isinstance(self.d.type, IntRegisterType)
        return (
            VariadicNeonRegArg(self.src_regs, self.arrangement),
            square_brackets_reg(self.d),
        )


ARM_NEON = Dialect(
    "arm_neon",
    [
        DSSFmlaVecScalarOp,
        DSSFMulVecScalarOp,
        DVarSSt1Op,
        GetRegisterOp,
    ],
    [
        NeonArrangementAttr,
        NEONRegisterType,
    ],
)
