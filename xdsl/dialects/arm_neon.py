from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.arm.assembly import reg, square_brackets_reg
from xdsl.dialects.arm.ops import ARMInstruction, ARMOperation
from xdsl.dialects.arm.registers import ARMRegisterType, IntRegisterType
from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
    VectorType,
    f16,
    f32,
    f64,
    i8,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    Operation,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
)
from xdsl.irdl import (
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
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

    @property
    def num_elements(self):
        return _NUM_ELEMENTS_BY_ARRANGEMENT[self.name]

    @staticmethod
    def from_vec_type(vec_type: VectorType):
        arrangement = _ARRANGEMENT_BY_TYPE.get(vec_type)
        if arrangement is None:
            raise ValueError(f"Invalid vector type for ARM NEON: {vec_type}")
        return arrangement


_NUM_ELEMENTS_BY_ARRANGEMENT = {"D": 2, "S": 4, "H": 8}
_ARRANGEMENT_BY_TYPE: dict[VectorType, NeonArrangement] = {
    VectorType(f16, (8,)): NeonArrangement.H,
    VectorType(f32, (4,)): NeonArrangement.S,
    VectorType(f64, (2,)): NeonArrangement.D,
}


@irdl_attr_definition
class NeonArrangementAttr(EnumAttribute[NeonArrangement], SpacedOpaqueSyntaxAttribute):
    """
    Attribute containing the arrangement specification.
    """

    name = "arm_neon.arrangement"


def vector_with_arrangement(
    reg: NEONRegisterType | SSAValue,
    arrangement: NeonArrangementAttr,
    *,
    index: int | None = None,
) -> str:
    if isinstance(reg, SSAValue):
        assert isinstance(reg.type, NEONRegisterType)
        reg = reg.type
    if index is None:
        return (
            f"{reg.register_name.data}."
            f"{arrangement.data.num_elements}"
            f"{arrangement.data.name}"
        )
    else:
        return f"{reg.register_name.data}.{arrangement.data.name}[{index}]"


def variadic_neon_reg_arg(
    regs: Sequence[SSAValue],
    arrangement: NeonArrangementAttr,
) -> str:
    """
    Returns the assembly string for a variadic NEON register argument with the given arrangement.
    """
    return (
        "{"
        + ", ".join(vector_with_arrangement(register, arrangement) for register in regs)
        + "}"
    )


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
class DSSFMulOp(ARMInstruction):
    """
    Floating-point multiply. Different instruction types supported:

    1. Vector:  multiplies corresponding floating-point values in the vectors in the two
    source NEON registers, and writes the result vector to the destination.

    See external [documentation](https://developer.arm.com/documentation/100069/0606/SIMD-Vector-Instructions/FMUL--vector-).

    2. Mixed: (first source operand is a vector, second is a scalar. Destination is a vector)
    This instruction multiplies each of the floating-point values in the first source
    operand by the second source operand and writes the resulting values to the corresponding
    lanes of the destination.
    Encoding: FMUL <Vd>.<T>, <Vn>.<T>, <Vm>.<idx>.

    See external [documentation](https://developer.arm.com/documentation/ddi0602/2024-12/SIMD-FP-Instructions/FMUL--vector---Floating-point-multiply--vector--?lang=en#T_option__4).
    """

    name = "arm_neon.dss.fmul"
    d = result_def(NEONRegisterType)
    s1 = operand_def(NEONRegisterType)
    s2 = operand_def(NEONRegisterType)
    scalar_idx = opt_prop_def(IntegerAttr[i8])
    arrangement = prop_def(NeonArrangementAttr)

    assembly_format = (
        "$s1 `,` $s2 (`[` $scalar_idx^ `]`)? $arrangement attr-dict "
        "`:` functional-type(operands, $d)"
    )

    def __init__(
        self,
        s1: Operation | SSAValue,
        s2: Operation | SSAValue,
        *,
        d: NEONRegisterType,
        scalar_idx: IntegerAttr | None,
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
            },
            properties={
                "scalar_idx": scalar_idx,
                "arrangement": arrangement,
            },
            result_types=(d,),
        )

    def assembly_line_args(self):
        return (
            vector_with_arrangement(self.d, self.arrangement),
            vector_with_arrangement(self.s1, self.arrangement),
            vector_with_arrangement(
                self.s2,
                self.arrangement,
                index=self.scalar_idx.value.data
                if self.scalar_idx is not None
                else None,
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

    SAME_NEON_REGISTER_TYPE: ClassVar = VarConstraint(
        "SAME_NEON_REGISTER_TYPE", base(NEONRegisterType)
    )

    name = "arm_neon.dss.fmla"
    res = result_def(SAME_NEON_REGISTER_TYPE)
    d = operand_def(SAME_NEON_REGISTER_TYPE)
    s1 = operand_def(NEONRegisterType)
    s2 = operand_def(NEONRegisterType)
    scalar_idx = prop_def(IntegerAttr[i8])
    arrangement = prop_def(NeonArrangementAttr)

    assembly_format = (
        "$d `,` $s1 `,` $s2 `[` $scalar_idx `]` $arrangement attr-dict `:` \
        `(` type($s1) `,` type($s2) `)` `->` type($res)"
    )

    def __init__(
        self,
        d: Operation | SSAValue,
        s1: Operation | SSAValue,
        s2: Operation | SSAValue,
        *,
        res: NEONRegisterType,
        scalar_idx: IntegerAttr,
        arrangement: NeonArrangement | NeonArrangementAttr,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(arrangement, NeonArrangement):
            arrangement = NeonArrangementAttr(arrangement)
        super().__init__(
            operands=(d, s1, s2),
            attributes={
                "comment": comment,
            },
            properties={
                "scalar_idx": scalar_idx,
                "arrangement": arrangement,
            },
            result_types=(res,),
        )

    def assembly_instruction_name(self) -> str:
        return "fmla"

    def assembly_line_args(self):
        return (
            vector_with_arrangement(self.res, self.arrangement),
            vector_with_arrangement(self.s1, self.arrangement),
            vector_with_arrangement(
                self.s2, self.arrangement, index=self.scalar_idx.value.data
            ),
        )


@irdl_op_definition
class DSDupOp(ARMInstruction):
    """
    Duplicate general-purpose register to vector.
    """

    name = "arm_neon.ds.dup"
    s = operand_def(IntRegisterType)
    d = result_def(NEONRegisterType)
    arrangement = prop_def(NeonArrangementAttr)

    assembly_format = "$s $arrangement attr-dict `:` type($s) `->` `(` type($d) `)`"

    def __init__(
        self,
        s: Operation | SSAValue,
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
            operands=(s,),
            attributes={
                "comment": comment,
                "arrangement": arrangement,
            },
            result_types=(d,),
        )

    def assembly_line_args(self):
        return (
            vector_with_arrangement(self.d, self.arrangement),
            reg(self.s),
        )


@irdl_op_definition
class DSVecMovOp(ARMInstruction):
    """
    Variant of MOV instruction which extracts a value from a specified lane in a NEON
    register into a general-purpose register.
    e.g. MOV X0, V3.S[1]
    Set X0 to the value of the second single word (bits 32-63) in V3.
    This instruction is an alias of UMOV.
    """

    name = "arm_neon.dsvec.mov"
    s = operand_def(NEONRegisterType)
    d = result_def(IntRegisterType)
    scalar_idx = prop_def(IntegerAttr[i8])
    arrangement = prop_def(NeonArrangementAttr)
    assembly_format = (
        "$s `[` $scalar_idx `]` $arrangement attr-dict `:` type($s) `->` type($d)"
    )

    def __init__(
        self,
        s: Operation | SSAValue,
        *,
        d: IntRegisterType,
        arrangement: NeonArrangement | NeonArrangementAttr,
        scalar_idx: IntegerAttr,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(arrangement, NeonArrangement):
            arrangement = NeonArrangementAttr(arrangement)
        super().__init__(
            operands=(s,),
            attributes={
                "comment": comment,
            },
            properties={
                "arrangement": arrangement,
                "scalar_idx": scalar_idx,
            },
            result_types=(d,),
        )

    def assembly_line_args(self):
        return (
            reg(self.d),
            vector_with_arrangement(
                self.s, self.arrangement, index=self.scalar_idx.value.data
            ),
        )


@irdl_op_definition
class DVarSLd1Op(ARMInstruction):
    """
    Neon structure load instruction reads data from memory into 64-bit Neon registers.
    LD1 loads data from memory into up to four registers, with no interleaving.
    """

    name = "arm_neon.dvars.ld1"
    s = operand_def(IntRegisterType)
    dest_regs = var_result_def(NEONRegisterType)
    arrangement = prop_def(NeonArrangementAttr)

    assembly_format = " ` ` `[` $s `]` $arrangement attr-dict `:` type($s) `->` `(` type($dest_regs) `)`"

    def __init__(
        self,
        s: Operation | SSAValue,
        result_types: Sequence[Attribute],
        *,
        arrangement: NeonArrangement | NeonArrangementAttr,
        comment: str | StringAttr | None = None,
    ):
        if not (1 <= len(self.dest_regs) <= 4):
            raise ValueError(
                f"dest_regs must contain between 1 and 4 elements, but got {len(self.dest_regs)}."
            )
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(arrangement, NeonArrangement):
            arrangement = NeonArrangementAttr(arrangement)
        super().__init__(
            operands=(s,),
            attributes={
                "comment": comment,
            },
            properties={
                "arrangement": arrangement,
            },
            result_types=[result_types],
        )

    def verify_(self) -> None:
        if not (1 <= len(self.dest_regs) <= 4):
            raise VerifyException(
                f"dest_regs must contain between 1 and 4 elements, but got {len(self.dest_regs)}."
            )

    def assembly_line_args(self):
        return (
            variadic_neon_reg_arg(self.dest_regs, self.arrangement),
            square_brackets_reg(self.s),
        )


@irdl_op_definition
class DVarSSt1Op(ARMInstruction):
    """
    Neon structure store instruction stores data from 64-bit Neon registers to memory.
    ST1 stores one to four registers of data to memory, with no interleaving.
    """

    name = "arm_neon.dvars.st1"
    d = operand_def(IntRegisterType)
    src_regs = var_operand_def(NEONRegisterType)
    arrangement = prop_def(NeonArrangementAttr)

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
            },
            properties={
                "arrangement": arrangement,
            },
            result_types=(d,),
        )

    def verify_(self) -> None:
        if not (1 <= len(self.src_regs) <= 4):
            raise VerifyException(
                f"src_regs must contain between 1 and 4 elements, but got {len(self.src_regs)}."
            )

    def assembly_line_args(self):
        return (
            variadic_neon_reg_arg(self.src_regs, self.arrangement),
            square_brackets_reg(self.d),
        )


ARM_NEON = Dialect(
    "arm_neon",
    [
        DSSFmlaVecScalarOp,
        DSSFMulOp,
        DSDupOp,
        DSVecMovOp,
        DVarSSt1Op,
        DVarSLd1Op,
        GetRegisterOp,
    ],
    [
        NeonArrangementAttr,
        NEONRegisterType,
    ],
)
