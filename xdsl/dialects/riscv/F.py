from abc import ABC
from dataclasses import dataclass, field

from xdsl.dialects.builtin import AnyIntegerAttr, StringAttr
from xdsl.ir import Data, Dialect, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from .base import Register, RegisterType
from .core import (
    AssemblyInstructionArgType,
    LabelAttr,
    RISCVInstruction,
    RISCVOp,
    SImm12Attr,
)

# region RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0


@dataclass(frozen=True)
class FloatRegister:
    """
    A RISC-V register.
    """

    name: str | None = field(default=None)
    """The register name. Should be one of `ABI_INDEX_BY_NAME` or `None`"""

    RV32F_INDEX_BY_NAME = {
        "ft0": 0,
        "ft1": 1,
        "ft2": 2,
        "ft3": 3,
        "ft4": 4,
        "ft5": 5,
        "ft6": 6,
        "ft7": 7,
        "fs0": 8,
        "fs1": 9,
        "fa0": 10,
        "fa1": 11,
        "fa2": 12,
        "fa3": 13,
        "fa4": 14,
        "fa5": 15,
        "fa6": 16,
        "fa7": 17,
        "fs2": 18,
        "fs3": 19,
        "fs4": 20,
        "fs5": 21,
        "fs6": 22,
        "fs7": 23,
        "fs8": 24,
        "fs9": 25,
        "fs10": 26,
        "fs11": 27,
        "ft8": 28,
        "ft9": 29,
        "ft10": 30,
        "ft11": 31,
    }


@irdl_attr_definition
class FloatRegisterType(Data[FloatRegister], TypeAttribute):
    """
    A RISC-V register type.
    """

    name = "riscv.freg"

    @property
    def register_name(self) -> str:
        """Returns name if allocated, raises ValueError if not"""
        if self.data.name is None:
            raise ValueError("Cannot get name for unallocated register")
        return self.data.name

    def print_assembly_instruction_arg(self) -> str:
        return self.register_name

    @staticmethod
    def parse_parameter(parser: Parser) -> FloatRegister:
        name = parser.parse_optional_identifier()
        if name is None:
            return FloatRegister()
        if not name.startswith("j"):
            assert name in FloatRegister.RV32F_INDEX_BY_NAME.keys()
        return FloatRegister(name)

    def print_parameter(self, printer: Printer) -> None:
        name = self.data.name
        if name is None:
            return
        printer.print_string(name)

    def verify(self) -> None:
        if self.data.name is None or self.data.name.startswith("j"):
            return
        if self.data.name not in FloatRegister.RV32F_INDEX_BY_NAME:
            raise VerifyException(
                f"{self.data.name} is not a valid float register name"
            )


class FloatRegisters(ABC):
    """Namespace for named register constants."""

    FT0 = FloatRegisterType(FloatRegister("ft0"))
    FT1 = FloatRegisterType(FloatRegister("ft1"))
    FT2 = FloatRegisterType(FloatRegister("ft2"))
    FT3 = FloatRegisterType(FloatRegister("ft3"))
    FT4 = FloatRegisterType(FloatRegister("ft4"))
    FT5 = FloatRegisterType(FloatRegister("ft5"))
    FT6 = FloatRegisterType(FloatRegister("ft6"))
    FT7 = FloatRegisterType(FloatRegister("ft7"))
    FS0 = FloatRegisterType(FloatRegister("fs0"))
    FS1 = FloatRegisterType(FloatRegister("fs1"))
    FA0 = FloatRegisterType(FloatRegister("fa0"))
    FA1 = FloatRegisterType(FloatRegister("fa1"))
    FA2 = FloatRegisterType(FloatRegister("fa2"))
    FA3 = FloatRegisterType(FloatRegister("fa3"))
    FA4 = FloatRegisterType(FloatRegister("fa4"))
    FA5 = FloatRegisterType(FloatRegister("fa5"))
    FA6 = FloatRegisterType(FloatRegister("fa6"))
    FA7 = FloatRegisterType(FloatRegister("fa7"))
    FS2 = FloatRegisterType(FloatRegister("fs2"))
    FS3 = FloatRegisterType(FloatRegister("fs3"))
    FS4 = FloatRegisterType(FloatRegister("fs4"))
    FS5 = FloatRegisterType(FloatRegister("fs5"))
    FS6 = FloatRegisterType(FloatRegister("fs6"))
    FS7 = FloatRegisterType(FloatRegister("fs7"))
    FS8 = FloatRegisterType(FloatRegister("fs8"))
    FS9 = FloatRegisterType(FloatRegister("fs9"))
    FS10 = FloatRegisterType(FloatRegister("fs10"))
    FS11 = FloatRegisterType(FloatRegister("fs11"))
    FT8 = FloatRegisterType(FloatRegister("ft8"))
    FT9 = FloatRegisterType(FloatRegister("ft9"))
    FT10 = FloatRegisterType(FloatRegister("ft10"))
    FT11 = FloatRegisterType(FloatRegister("ft11"))


class RdRsRsRsFloatOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that take three
    floating-point input registers and a destination register,
    e.g: fused-multiply-add (FMA) instructions.
    """

    rd: OpResult = result_def(FloatRegisterType)
    rs1: Operand = operand_def(FloatRegisterType)
    rs2: Operand = operand_def(FloatRegisterType)
    rs3: Operand = operand_def(FloatRegisterType)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        rs3: Operation | SSAValue,
        *,
        rd: FloatRegisterType | FloatRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = FloatRegisterType(FloatRegister())
        elif isinstance(rd, FloatRegister):
            rd = FloatRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2, rs3],
            attributes={
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.rs1, self.rs2, self.rs3


class RdRsRsFloatOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that
    take two floating-point input registers and a destination.
    """

    rd: OpResult = result_def(FloatRegisterType)
    rs1: Operand = operand_def(FloatRegisterType)
    rs2: Operand = operand_def(FloatRegisterType)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: FloatRegisterType | FloatRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = FloatRegisterType(FloatRegister())
        elif isinstance(rd, FloatRegister):
            rd = FloatRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.rs1, self.rs2


class RdRsRsFloatFloatIntegerOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that take
    two floating-point input registers and an integer destination register.
    """

    rd: OpResult = result_def(RegisterType)
    rs1: Operand = operand_def(FloatRegisterType)
    rs2: Operand = operand_def(FloatRegisterType)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: RegisterType | Register | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.rs1, self.rs2


class RdRsFloatOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that take a floating-point
    input register and a floating destination register.
    """

    rd: OpResult = result_def(FloatRegisterType)
    rs: Operand = operand_def(FloatRegisterType)

    def __init__(
        self,
        rs: Operation | SSAValue,
        *,
        rd: FloatRegisterType | FloatRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = FloatRegisterType(FloatRegister())
        elif isinstance(rd, FloatRegister):
            rd = FloatRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs],
            result_types=[rd],
            attributes={"comment": comment},
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.rs


class RdRsFloatIntegerOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that take a floating-point
    input register and an integer destination register.
    """

    rd: OpResult = result_def(RegisterType)
    rs: Operand = operand_def(FloatRegisterType)

    def __init__(
        self,
        rs: Operation | SSAValue,
        *,
        rd: RegisterType | Register | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs],
            result_types=[rd],
            attributes={"comment": comment},
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.rs


class RdRsIntegerFloatOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that take an integer
    input register and a floating-point destination register.
    """

    rd: OpResult = result_def(FloatRegisterType)
    rs: Operand = operand_def(RegisterType)

    def __init__(
        self,
        rs: Operation | SSAValue,
        *,
        rd: FloatRegisterType | FloatRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = FloatRegisterType(FloatRegister())
        elif isinstance(rd, FloatRegister):
            rd = FloatRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs],
            result_types=[rd],
            attributes={"comment": comment},
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.rs


class RsRsImmFloatOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that have two source registers
    (one integer and one floating-point) and an immediate.
    """

    rs1: Operand = operand_def(RegisterType)
    rs2: Operand = operand_def(FloatRegisterType)
    immediate: AnyIntegerAttr = attr_def(AnyIntegerAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        immediate: int | AnyIntegerAttr | str | LabelAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = SImm12Attr(immediate)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rs1, self.rs2, self.immediate


class RdRsImmFloatOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RV32Foperations that have one floating-point
    destination register, one source register and
    one immediate operand.
    """

    rd: OpResult = result_def(FloatRegisterType)
    rs1: Operand = operand_def(RegisterType)
    immediate: AnyIntegerAttr | LabelAttr = attr_def(AnyIntegerAttr | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | AnyIntegerAttr | str | LabelAttr,
        *,
        rd: FloatRegisterType | FloatRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = SImm12Attr(immediate)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)

        if rd is None:
            rd = FloatRegisterType(FloatRegister())
        elif isinstance(rd, FloatRegister):
            rd = FloatRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs1],
            result_types=[rd],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.rs1, self.immediate


@irdl_op_definition
class FMAddSOp(RdRsRsRsFloatOperation):
    """
    Perform single-precision fused multiply addition.

    f[rd] = f[rs1]×f[rs2]+f[rs3]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmadd-s
    """

    name = "riscv.fmadd.s"


@irdl_op_definition
class FMSubSOp(RdRsRsRsFloatOperation):
    """
    Perform single-precision fused multiply substraction.

    f[rd] = f[rs1]×f[rs2]+f[rs3]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmsub-s
    """

    name = "riscv.fmsub.s"


@irdl_op_definition
class FNMSubSOp(RdRsRsRsFloatOperation):
    """
    Perform single-precision fused multiply substraction.

    f[rd] = -f[rs1]×f[rs2]+f[rs3]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fnmsub-s
    """

    name = "riscv.fnmsub.s"


@irdl_op_definition
class FNMAddSOp(RdRsRsRsFloatOperation):
    """
    Perform single-precision fused multiply addition.

    f[rd] = -f[rs1]×f[rs2]-f[rs3]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fnmadd-s
    """

    name = "riscv.fnmadd.s"


@irdl_op_definition
class FAddSOp(RdRsRsFloatOperation):
    """
    Perform single-precision floating-point addition.

    f[rd] = f[rs1]+f[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fadd-s
    """

    name = "riscv.fadd.s"


@irdl_op_definition
class FSubSOp(RdRsRsFloatOperation):
    """
    Perform single-precision floating-point substraction.

    f[rd] = f[rs1]-f[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsub-s
    """

    name = "riscv.fsub.s"


@irdl_op_definition
class FMulSOp(RdRsRsFloatOperation):
    """
    Perform single-precision floating-point multiplication.

    f[rd] = f[rs1]×f[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmul-s
    """

    name = "riscv.fmul.s"


@irdl_op_definition
class FDivSOp(RdRsRsFloatOperation):
    """
    Perform single-precision floating-point division.

    f[rd] = f[rs1] / f[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fdiv-s
    """

    name = "riscv.fdiv.s"


@irdl_op_definition
class FSqrtSOp(RdRsFloatOperation):
    """
    Perform single-precision floating-point square root.

    f[rd] = sqrt(f[rs1])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsqrt-s
    """

    name = "riscv.fsqrt.s"


@irdl_op_definition
class FSgnJSOp(RdRsRsFloatOperation):
    """
    Produce a result that takes all bits except the sign bit from rs1.
    The result’s sign bit is rs2’s sign bit.

    f[rd] = {f[rs2][31], f[rs1][30:0]}

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsgnj.s
    """

    name = "riscv.fsgnj.s"


@irdl_op_definition
class FSgnJNSOp(RdRsRsFloatOperation):
    """
    Produce a result that takes all bits except the sign bit from rs1.
    The result’s sign bit is opposite of rs2’s sign bit.


    f[rd] = {~f[rs2][31], f[rs1][30:0]}

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsgnjn.s
    """

    name = "riscv.fsgnjn.s"


@irdl_op_definition
class FSgnJXSOp(RdRsRsFloatOperation):
    """
    Produce a result that takes all bits except the sign bit from rs1.
    The result’s sign bit is XOR of sign bit of rs1 and rs2.

    f[rd] = {f[rs1][31] ^ f[rs2][31], f[rs1][30:0]}

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsgnjx.s
    """

    name = "riscv.fsgnjx.s"


@irdl_op_definition
class FMinSOp(RdRsRsFloatOperation):
    """
    Write the smaller of single precision data in rs1 and rs2 to rd.

    f[rd] = min(f[rs1], f[rs2])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmin-s
    """

    name = "riscv.fmin.s"


@irdl_op_definition
class FMaxSOp(RdRsRsFloatOperation):
    """
    Write the larger of single precision data in rs1 and rs2 to rd.

    f[rd] = max(f[rs1], f[rs2])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmax-s
    """

    name = "riscv.fmax.s"


@irdl_op_definition
class FCvtWSOp(RdRsFloatIntegerOperation):
    """
    Convert a floating-point number in floating-point register rs1 to a signed 32-bit in integer register rd.

    x[rd] = sext(s32_{f32}(f[rs1]))

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt.w.s
    """

    name = "riscv.fcvt.w.s"


@irdl_op_definition
class FCvtWuSOp(RdRsFloatIntegerOperation):
    """
    Convert a floating-point number in floating-point register rs1 to a signed 32-bit in unsigned integer register rd.

    x[rd] = sext(u32_{f32}(f[rs1]))

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt.wu.s
    """

    name = "riscv.fcvt.wu.s"


@irdl_op_definition
class FMvXWOp(RdRsFloatIntegerOperation):
    """
    Move the single-precision value in floating-point register rs1 represented in IEEE 754-2008 encoding to the lower 32 bits of integer register rd.

    x[rd] = sext(f[rs1][31:0])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmv.x.w
    """

    name = "riscv.fmv.x.w"


@irdl_op_definition
class FeqSOP(RdRsRsFloatFloatIntegerOperation):
    """
    Performs a quiet equal comparison between floating-point registers rs1 and rs2 and record the Boolean result in integer register rd.
    Only signaling NaN inputs cause an Invalid Operation exception.
    The result is 0 if either operand is NaN.

    x[rd] = f[rs1] == f[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#feq.s
    """

    name = "riscv.feq.s"


@irdl_op_definition
class FltSOP(RdRsRsFloatFloatIntegerOperation):
    """
    Performs a quiet less comparison between floating-point registers rs1 and rs2 and record the Boolean result in integer register rd.
    Only signaling NaN inputs cause an Invalid Operation exception.
    The result is 0 if either operand is NaN.

    x[rd] = f[rs1] < f[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#flt.s
    """

    name = "riscv.flt.s"


@irdl_op_definition
class FleSOP(RdRsRsFloatFloatIntegerOperation):
    """
    Performs a quiet less or equal comparison between floating-point registers rs1 and rs2 and record the Boolean result in integer register rd.
    Only signaling NaN inputs cause an Invalid Operation exception.
    The result is 0 if either operand is NaN.

    x[rd] = f[rs1] <= f[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fle.s
    """

    name = "riscv.fle.s"


@irdl_op_definition
class FClassSOp(RdRsFloatIntegerOperation):
    """
    Examines the value in floating-point register rs1 and writes to integer register rd a 10-bit mask that indicates the class of the floating-point number.
    The format of the mask is described in [classify table]_.
    The corresponding bit in rd will be set if the property is true and clear otherwise.
    All other bits in rd are cleared. Note that exactly one bit in rd will be set.

    x[rd] = classifys(f[rs1])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fclass.s
    """

    name = "riscv.fclass.s"


@irdl_op_definition
class FCvtSWOp(RdRsIntegerFloatOperation):
    """
    Converts a 32-bit signed integer, in integer register rs1 into a floating-point number in floating-point register rd.

    f[rd] = f32_{s32}(x[rs1])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt.s.w
    """

    name = "riscv.fcvt.s.w"


@irdl_op_definition
class FCvtSWuOp(RdRsIntegerFloatOperation):
    """
    Converts a 32-bit unsigned integer, in integer register rs1 into a floating-point number in floating-point register rd.

    f[rd] = f32_{u32}(x[rs1])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt.s.wu
    """

    name = "riscv.fcvt.s.wu"


@irdl_op_definition
class FMvWXOp(RdRsIntegerFloatOperation):
    """
    Move the single-precision value encoded in IEEE 754-2008 standard encoding from the lower 32 bits of integer register rs1 to the floating-point register rd.

    f[rd] = x[rs1][31:0]


    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmv.w.x
    """

    name = "riscv.fmv.w.x"


@irdl_op_definition
class FLwOp(RdRsImmFloatOperation):
    """
    Load a single-precision value from memory into floating-point register rd.

    f[rd] = M[x[rs1] + sext(offset)][31:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#flw
    """

    name = "riscv.flw"


@irdl_op_definition
class FSwOp(RsRsImmFloatOperation):
    """
    Store a single-precision value from floating-point register rs2 to memory.

    M[x[rs1] + offset] = f[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsw
    """

    name = "riscv.fsw"


# endregion

# region RISC-V SSA Helpers


@irdl_op_definition
class GetFloatRegisterOp(IRDLOperation, RISCVOp):
    """
    This instruction allows us to create an SSAValue with for a given floating register name. This
    is useful for bridging the RISC-V convention that stores the result of function calls
    in `a0` and `a1` into SSA form.
    """

    name = "riscv.get_float_register"
    res: OpResult = result_def(FloatRegisterType)

    def __init__(
        self,
        register_type: FloatRegisterType | FloatRegister,
    ):
        if isinstance(register_type, FloatRegister):
            register_type = FloatRegisterType(register_type)
        super().__init__(result_types=[register_type])

    def assembly_line(self) -> str | None:
        # Don't print assembly for creating a SSA value representing register
        return None


# endregion

RISCV_F = Dialect(
    [
        FMAddSOp,
        FMSubSOp,
        FNMSubSOp,
        FNMAddSOp,
        FAddSOp,
        FSubSOp,
        FMulSOp,
        FDivSOp,
        FSqrtSOp,
        FSgnJSOp,
        FSgnJNSOp,
        FSgnJXSOp,
        FMinSOp,
        FMaxSOp,
        FCvtWSOp,
        FCvtWuSOp,
        FMvXWOp,
        FeqSOP,
        FltSOP,
        FleSOP,
        FClassSOp,
        FCvtSWOp,
        FCvtSWuOp,
        FMvWXOp,
        FLwOp,
        FSwOp,
        GetFloatRegisterOp,
    ],
    [
        FloatRegisterType,
    ],
)
