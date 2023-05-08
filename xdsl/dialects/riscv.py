from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Annotated

from xdsl.ir import (
    Dialect,
    Operation,
    SSAValue,
    Data,
    OpResult,
    TypeAttribute,
)

from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    irdl_attr_definition,
    Operand,
    OpAttr,
    OptOpAttr,
)

from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    UnitAttr,
    IntegerAttr,
)
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True)
class Register:
    """
    A RISC-V register.
    """

    name: str | None = field(default=None)
    """The register name. Should be one of `ABI_INDEX_BY_NAME` or `None`"""

    ABI_INDEX_BY_NAME = {
        "zero": 0,
        "ra": 1,
        "sp": 2,
        "gp": 3,
        "tp": 4,
        "t0": 5,
        "t1": 6,
        "t2": 7,
        "fp": 8,
        "s0": 8,
        "s1": 9,
        "a0": 10,
        "a1": 11,
        "a2": 12,
        "a3": 13,
        "a4": 14,
        "a5": 15,
        "a6": 16,
        "a7": 17,
        "s2": 18,
        "s3": 19,
        "s4": 20,
        "s5": 21,
        "s6": 22,
        "s7": 23,
        "s8": 24,
        "s9": 25,
        "s10": 26,
        "s11": 27,
        "t3": 28,
        "t4": 29,
        "t5": 30,
        "t6": 31,
    }


class Registers(ABC):
    """Namespace for named register constants."""

    ZERO = Register("zero")
    RA = Register("ra")
    SP = Register("sp")
    GP = Register("gp")
    TP = Register("tp")
    T0 = Register("t0")
    T1 = Register("t1")
    T2 = Register("t2")
    FP = Register("fp")
    S0 = Register("s0")
    S1 = Register("s1")
    A0 = Register("a0")
    A1 = Register("a1")
    A2 = Register("a2")
    A3 = Register("a3")
    A4 = Register("a4")
    A5 = Register("a5")
    A6 = Register("a6")
    A7 = Register("a7")
    S2 = Register("s2")
    S3 = Register("s3")
    S4 = Register("s4")
    S5 = Register("s5")
    S6 = Register("s6")
    S7 = Register("s7")
    S8 = Register("s8")
    S9 = Register("s9")
    S10 = Register("s10")
    S11 = Register("s11")
    T3 = Register("t3")
    T4 = Register("t4")
    T5 = Register("t5")
    T6 = Register("t6")


@irdl_attr_definition
class RegisterType(Data[Register], TypeAttribute):
    """
    A RISC-V register type.
    """

    name = "riscv.reg"

    @staticmethod
    def parse_parameter(parser: Parser) -> Register:
        name = parser.try_parse_bare_id()
        if name is None:
            return Register()
        text = name.text
        return Register(text)

    def print_parameter(self, printer: Printer) -> None:
        name = self.data.name
        if name is None:
            return
        printer.print_string(name)

    @property
    def abi_name(self):
        return self.data.name


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "riscv.label"

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string_literal(self.data)


class RISCVOp(Operation, ABC):
    pass


# region Base Operation classes


class RdRsRsOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations that have one destination register, and two source
    registers.

    This is called R-Type in the RISC-V specification.
    """

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: RegisterType | Register | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)

        super().__init__(
            operands=[rs1, rs2],
            result_types=[rd],
        )


class RdImmOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations that have one destination register, and one
    immediate operand (e.g. U-Type and J-Type instructions in the RISC-V spec).
    """

    rd: Annotated[OpResult, RegisterType]
    immediate: OpAttr[AnyIntegerAttr | LabelAttr]

    def __init__(
        self,
        immediate: int | AnyIntegerAttr | str | LabelAttr,
        *,
        rd: RegisterType | Register | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr.from_int_and_width(immediate, 32)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        super().__init__(
            attributes={
                "immediate": immediate,
            },
            result_types=[rd],
        )


class RdRsImmOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations that have one destination register, one source
    register and one immediate operand.

    This is called I-Type in the RISC-V specification.
    """

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    immediate: OpAttr[AnyIntegerAttr | LabelAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | AnyIntegerAttr | str | LabelAttr,
        *,
        rd: RegisterType | Register | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, 32)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)

        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        super().__init__(
            operands=[rs1],
            attributes={
                "immediate": immediate,
            },
            result_types=[rd],
        )


class RdRsOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V pseudo-instructions that have one destination register and one
    source register.
    """

    rd: Annotated[OpResult, RegisterType]
    rs: Annotated[Operand, RegisterType]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        *,
        rd: RegisterType | Register | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        super().__init__(
            operands=[rs1],
            result_types=[rd],
        )


class RsRsOffOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations that have one source register and a destination
    register, and an offset.

    This is called B-Type in the RISC-V specification.
    """

    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    offset: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        offset: AnyIntegerAttr,
    ):
        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "offset": offset,
            },
        )


class RsRsImmOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations that have two source registers and an
    immediate.

    This is called S-Type in the RISC-V specification.
    """

    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    immediate: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        immediate: AnyIntegerAttr,
    ):
        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "immediate": immediate,
            },
        )


class NullaryOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations that have neither sources nor destinations.
    """

    def __init__(self):
        super().__init__()


class CsrReadWriteOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations performing a swap to/from a CSR.

    The 'writeonly' attribute controls the actual behaviour of the operation:
    * when True, the operation writes the rs value to the CSR but never reads it and
      in this case rd *must* be allocated to x0
    * when False, a proper atomic swap is performed and the previous CSR value is
      returned in rd
    """

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    csr: OpAttr[AnyIntegerAttr]
    writeonly: OptOpAttr[UnitAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        csr: AnyIntegerAttr,
        *,
        writeonly: bool = False,
        rd: RegisterType | Register | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        super().__init__(
            operands=[rs1],
            attributes={
                "csr": csr,
                "writeonly": UnitAttr() if writeonly else None,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if not self.writeonly:
            return
        if not isinstance(self.rd.typ, RegisterType):
            return
        if self.rd.typ.abi_name is not None and self.rd.typ.abi_name != "zero":
            raise VerifyException(
                "When in 'writeonly' mode, destination must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rd.typ.abi_name}'"
            )


class CsrBitwiseOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations performing a masked bitwise operation on the
    CSR while returning the original value.

    The 'readonly' attribute controls the actual behaviour of the operation:
    * when True, the operation is guaranteed to have no side effects that can
      be potentially related to writing to a CSR; in this case rs *must be
      allocated to x0*
    * when False, the bitwise operations is performed and any side effect related
      to writing to a CSR takes place even if the mask in rs has no actual bits set.
    """

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    csr: OpAttr[AnyIntegerAttr]
    readonly: OptOpAttr[UnitAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        csr: AnyIntegerAttr,
        *,
        readonly: bool = False,
        rd: RegisterType | Register | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        super().__init__(
            operands=[rs1],
            attributes={
                "csr": csr,
                "readonly": UnitAttr() if readonly else None,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if not self.readonly:
            return
        if not isinstance(self.rs1.typ, RegisterType):
            return
        if self.rs1.typ.abi_name is not None and self.rs1.typ.abi_name != "zero":
            raise VerifyException(
                "When in 'readonly' mode, source must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rs1.typ.abi_name}'"
            )


class CsrReadWriteImmOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations performing a write immediate to/read from a CSR.

    The 'writeonly' attribute controls the actual behaviour of the operation:
    * when True, the operation writes the rs value to the CSR but never reads it and
      in this case rd *must* be allocated to x0
    * when False, a proper atomic swap is performed and the previous CSR value is
      returned in rd
    """

    rd: Annotated[OpResult, RegisterType]
    csr: OpAttr[AnyIntegerAttr]
    writeonly: OptOpAttr[UnitAttr]
    immediate: OptOpAttr[AnyIntegerAttr]

    def __init__(
        self,
        csr: AnyIntegerAttr,
        immediate: AnyIntegerAttr,
        *,
        writeonly: bool = False,
        rd: RegisterType | Register | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        super().__init__(
            attributes={
                "csr": csr,
                "immediate": immediate,
                "writeonly": UnitAttr() if writeonly else None,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if self.writeonly is None:
            return
        if not isinstance(self.rd.typ, RegisterType):
            return
        if self.rd.typ.abi_name is not None and self.rd.typ.abi_name != "zero":
            raise VerifyException(
                "When in 'writeonly' mode, destination must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rd.typ.abi_name}'"
            )


class CsrBitwiseImmOperation(IRDLOperation, RISCVOp, ABC):
    """
    A base class for RISC-V operations performing a masked bitwise operation on the
    CSR while returning the original value. The bitmask is specified in the 'immediate'
    attribute.

    The 'immediate' attribute controls the actual behaviour of the operation:
    * when equals to zero, the operation is guaranteed to have no side effects
      that can be potentially related to writing to a CSR;
    * when not equal to zero, any side effect related to writing to a CSR takes
      place.
    """

    rd: Annotated[OpResult, RegisterType]
    csr: OpAttr[AnyIntegerAttr]
    immediate: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        csr: AnyIntegerAttr,
        immediate: AnyIntegerAttr,
        *,
        rd: RegisterType | Register | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, Register):
            rd = RegisterType(rd)
        super().__init__(
            attributes={
                "csr": csr,
                "immediate": immediate,
            },
            result_types=[rd],
        )


# endregion

# region RV32I/RV64I: 2.4 Integer Computational Instructions

## Integer Register-Immediate Instructions


@irdl_op_definition
class AddiOp(RdRsImmOperation):
    """
    Adds the sign-extended 12-bit immediate to register rs1.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] + sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#addi
    """

    name = "riscv.addi"


@irdl_op_definition
class SltiOp(RdRsImmOperation):
    """
    Place the value 1 in register rd if register rs1 is less than the sign-extended
    immediate when both are treated as signed numbers, else 0 is written to rd.

    x[rd] = x[rs1] <s sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slti
    """

    name = "riscv.slti"


@irdl_op_definition
class SltiuOp(RdRsImmOperation):
    """
    Place the value 1 in register rd if register rs1 is less than the immediate when
    both are treated as unsigned numbers, else 0 is written to rd.

    x[rd] = x[rs1] <u sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sltiu
    """

    name = "riscv.sltiu"


@irdl_op_definition
class AndiOp(RdRsImmOperation):
    """
    Performs bitwise AND on register rs1 and the sign-extended 12-bit
    immediate and place the result in rd.

    x[rd] = x[rs1] & sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#andi
    """

    name = "riscv.andi"


@irdl_op_definition
class OriOp(RdRsImmOperation):
    """
    Performs bitwise OR on register rs1 and the sign-extended 12-bit immediate and place
    the result in rd.

    x[rd] = x[rs1] | sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#ori
    """

    name = "riscv.ori"


@irdl_op_definition
class XoriOp(RdRsImmOperation):
    """
    Performs bitwise XOR on register rs1 and the sign-extended 12-bit immediate and place
    the result in rd.

    x[rd] = x[rs1] ^ sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#xori
    """

    name = "riscv.xori"


@irdl_op_definition
class SlliOp(RdRsImmOperation):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] << shamt

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slli
    """

    name = "riscv.slli"


@irdl_op_definition
class SrliOp(RdRsImmOperation):
    """
    Performs logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>u shamt

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srli
    """

    name = "riscv.srli"


@irdl_op_definition
class SraiOp(RdRsImmOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>s shamt

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai
    """

    name = "riscv.srai"


@irdl_op_definition
class LuiOp(RdImmOperation):
    """
    Build 32-bit constants and uses the U-type format. LUI places the U-immediate value
    in the top 20 bits of the destination register rd, filling in the lowest 12 bits with zeros.

    x[rd] = sext(immediate[31:12] << 12)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lui
    """

    name = "riscv.lui"


@irdl_op_definition
class AuipcOp(RdImmOperation):
    """
    Build pc-relative addresses and uses the U-type format. AUIPC forms a 32-bit offset
    from the 20-bit U-immediate, filling in the lowest 12 bits with zeros, adds this
    offset to the pc, then places the result in register rd.

    x[rd] = pc + sext(immediate[31:12] << 12)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#auipc
    """

    name = "riscv.auipc"


@irdl_op_definition
class MVOp(RdRsOperation):
    """
    A pseudo instruction to copy contents of one register to another.

    Equivalent to `addi rd, rs, 0`
    """

    name = "riscv.mv"


## Integer Register-Register Operations


@irdl_op_definition
class AddOp(RdRsRsOperation):
    """
    Adds the registers rs1 and rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] + x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#add
    """

    name = "riscv.add"


@irdl_op_definition
class SltOp(RdRsRsOperation):
    """
    Place the value 1 in register rd if register rs1 is less than register rs2 when both
    are treated as signed numbers, else 0 is written to rd.

    x[rd] = x[rs1] <s x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slt
    """

    name = "riscv.slt"


@irdl_op_definition
class SltuOp(RdRsRsOperation):
    """
    Place the value 1 in register rd if register rs1 is less than register rs2 when both
    are treated as unsigned numbers, else 0 is written to rd.

    x[rd] = x[rs1] <u x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sltu
    """

    name = "riscv.sltu"


@irdl_op_definition
class AndOp(RdRsRsOperation):
    """
    Performs bitwise AND on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] & x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#and
    """

    name = "riscv.and"


@irdl_op_definition
class OrOp(RdRsRsOperation):
    """
    Performs bitwise OR on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] | x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#or
    """

    name = "riscv.or"


@irdl_op_definition
class XorOp(RdRsRsOperation):
    """
    Performs bitwise XOR on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] ^ x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#xor
    """

    name = "riscv.xor"


@irdl_op_definition
class SllOp(RdRsRsOperation):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of register rs2.

    x[rd] = x[rs1] << x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sll
    """

    name = "riscv.sll"


@irdl_op_definition
class SrlOp(RdRsRsOperation):
    """
    Logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of register rs2.

    x[rd] = x[rs1] >>u x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srl
    """

    name = "riscv.srl"


@irdl_op_definition
class SubOp(RdRsRsOperation):
    """
    Subtracts the registers rs1 and rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] - x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sub
    """

    name = "riscv.sub"


@irdl_op_definition
class SraOp(RdRsRsOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of register rs2.

    x[rd] = x[rs1] >>s x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sub
    """

    name = "riscv.sra"


@irdl_op_definition
class NopOp(NullaryOperation):
    """
    Does not change any user-visible state, except for advancing the pc register.
    Canonical nop is encoded as addi x0, x0, 0.
    """

    name = "riscv.nop"


# endregion

# region RV32I/RV64I: 2.5 Control Transfer Instructions

# Unconditional jumps


@irdl_op_definition
class JalOp(RdImmOperation):
    """
    Jump to address and place return address in rd.

    jal mylabel is a pseudoinstruction for jal ra, mylabel

    x[rd] = pc+4; pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#jal
    """

    name = "riscv.jal"


@irdl_op_definition
class JOp(RdImmOperation):
    """
    A pseudo-instruction, for unconditional jumps you don't expect to return from.
    Is equivalent to JalOp with `rd` = `x0`.
    Used to be a part of the spec, removed in 2.0.
    """

    name = "riscv.j"

    def __init__(self, immediate: int | AnyIntegerAttr | str | LabelAttr):
        super().__init__(immediate, rd=Registers.ZERO)


@irdl_op_definition
class JalrOp(RdRsImmOperation):
    """
    Jump to address and place return address in rd.

    ```
    t = pc+4
    pc = (x[rs1] + sext(offset)) & ~1
    x[rd] = t
    ```

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#jalr
    """

    name = "riscv.jalr"


@irdl_op_definition
class ReturnOp(NullaryOperation):
    """
    Pseudo-op for returning from subroutine.

    Equivalent to `jalr x0, x1, 0`
    """

    name = "riscv.ret"

    def __init__(self):
        super().__init__()


# Conditional Branches


@irdl_op_definition
class BeqOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 and rs2 are equal.

    if (x[rs1] == x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#beq
    """

    name = "riscv.beq"


@irdl_op_definition
class BneOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 and rs2 are not equal.

    if (x[rs1] != x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bne
    """

    name = "riscv.bne"


@irdl_op_definition
class BltOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 is less than rs2, using signed comparison.

    if (x[rs1] <s x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#blt
    """

    name = "riscv.blt"


@irdl_op_definition
class BgeOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using signed comparison.

    if (x[rs1] >=s x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bge
    """

    name = "riscv.bge"


@irdl_op_definition
class BltuOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 is less than rs2, using unsigned comparison.

    if (x[rs1] <u x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bltu
    """

    name = "riscv.bltu"


@irdl_op_definition
class BgeuOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using unsigned comparison.

    if (x[rs1] >=u x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bgeu
    """

    name = "riscv.bgeu"


# endregion

# region RV32I/RV64I: 2.6 Load and Store Instructions


@irdl_op_definition
class LbOp(RdRsImmOperation):
    """
    Loads a 8-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = sext(M[x[rs1] + sext(offset)][7:0])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lb
    """

    name = "riscv.lb"


@irdl_op_definition
class LbuOp(RdRsImmOperation):
    """
    Loads a 8-bit value from memory and zero-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = M[x[rs1] + sext(offset)][7:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lbu
    """

    name = "riscv.lbu"


@irdl_op_definition
class LhOp(RdRsImmOperation):
    """
    Loads a 16-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = sext(M[x[rs1] + sext(offset)][15:0])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lh
    """

    name = "riscv.lh"


@irdl_op_definition
class LhuOp(RdRsImmOperation):
    """
    Loads a 16-bit value from memory and zero-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = M[x[rs1] + sext(offset)][15:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lhu
    """

    name = "riscv.lhu"


@irdl_op_definition
class LwOp(RdRsImmOperation):
    """
    Loads a 32-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = sext(M[x[rs1] + sext(offset)][31:0])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lw
    """

    name = "riscv.lw"


@irdl_op_definition
class SbOp(RsRsImmOperation):
    """
    Store 8-bit, values from the low bits of register rs2 to memory.

    M[x[rs1] + sext(offset)] = x[rs2][7:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sb
    """

    name = "riscv.sb"


@irdl_op_definition
class ShOp(RsRsImmOperation):
    """
    Store 16-bit, values from the low bits of register rs2 to memory.

    M[x[rs1] + sext(offset)] = x[rs2][15:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sh

    """

    name = "riscv.sh"


@irdl_op_definition
class SwOp(RsRsImmOperation):
    """
    Store 32-bit, values from the low bits of register rs2 to memory.

    M[x[rs1] + sext(offset)] = x[rs2][31:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sw
    """

    name = "riscv.sw"


# endregion

# region RV32I/RV64I: 2.8 Control and Status Register Instructions


@irdl_op_definition
class CsrrwOp(CsrReadWriteOperation):
    """
    Atomically swaps values in the CSRs and integer registers.
    CSRRW reads the old value of the CSR, zero-extends the value to XLEN bits,
    then writes it to integer register rd. The initial value in rs1 is written
    to the CSR. If the 'writeonly' attribute evaluates to False, then the
    instruction shall not read the CSR and shall not cause any of the side effects
    that might occur on a CSR read; in this case rd *must be allocated to x0*.

    t = CSRs[csr]; CSRs[csr] = x[rs1]; x[rd] = t

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrw
    """

    name = "riscv.csrrw"


@irdl_op_definition
class CsrrsOp(CsrBitwiseOperation):
    """
    Reads the value of the CSR, zero-extends the value to XLEN bits, and writes
    it to integer register rd. The initial value in integer register rs1 is treated
    as a bit mask that specifies bit positions to be set in the CSR.
    Any bit that is high in rs1 will cause the corresponding bit to be set in the CSR,
    if that CSR bit is writable. Other bits in the CSR are unaffected (though CSRs might
    have side effects when written).

    If the 'readonly' attribute evaluates to True, then the instruction will not write
    to the CSR at all, and so shall not cause any of the side effects that might otherwise
    occur on a CSR write, such as raising illegal instruction exceptions on accesses to
    read-only CSRs. Note that if rs1 specifies a register holding a zero value other than x0,
    the instruction will still attempt to write the unmodified value back to the CSR and will
    cause any attendant side effects.

    t = CSRs[csr]; CSRs[csr] = t | x[rs1]; x[rd] = t

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrs
    """

    name = "riscv.csrrs"


@irdl_op_definition
class CsrrcOp(CsrBitwiseOperation):
    """
    Reads the value of the CSR, zero-extends the value to XLEN bits, and writes
    it to integer register rd. The initial value in integer register rs1 is treated
    as a bit mask that specifies bit positions to be cleared in the CSR.
    Any bit that is high in rs1 will cause the corresponding bit to be cleared in the CSR,
    if that CSR bit is writable. Other bits in the CSR are unaffected (though CSRs might
    have side effects when written).

    If the 'readonly' attribute evaluates to True, then the instruction will not write
    to the CSR at all, and so shall not cause any of the side effects that might otherwise
    occur on a CSR write, such as raising illegal instruction exceptions on accesses to
    read-only CSRs. Note that if rs1 specifies a register holding a zero value other than x0,
    the instruction will still attempt to write the unmodified value back to the CSR and will
    cause any attendant side effects.

    t = CSRs[csr]; CSRs[csr] = t &~x[rs1]; x[rd] = t

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrc
    """

    name = "riscv.csrrc"


@irdl_op_definition
class CsrrwiOp(CsrReadWriteImmOperation):
    """
    Update the CSR using an XLEN-bit value obtained by zero-extending the
    'immediate' attribute.
    If the 'writeonly' attribute evaluates to False, then the
    instruction shall not read the CSR and shall not cause any of the side effects
    that might occur on a CSR read; in this case rd *must be allocated to x0*.

    x[rd] = CSRs[csr]; CSRs[csr] = zimm

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrwi
    """

    name = "riscv.csrrwi"


@irdl_op_definition
class CsrrsiOp(CsrBitwiseImmOperation):
    """
    Reads the value of the CSR, zero-extends the value to XLEN bits, and writes
    it to integer register rd. The value in the 'immediate' attribute is treated
    as a bit mask that specifies bit positions to be set in the CSR.
    Any bit that is high in it will cause the corresponding bit to be set in the CSR,
    if that CSR bit is writable. Other bits in the CSR are unaffected (though CSRs might
    have side effects when written).

    If the 'immediate' attribute value is zero, then the instruction will not write
    to the CSR at all, and so shall not cause any of the side effects that might otherwise
    occur on a CSR write, such as raising illegal instruction exceptions on accesses to
    read-only CSRs.

    t = CSRs[csr]; CSRs[csr] = t | zimm; x[rd] = t

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrsi
    """

    name = "riscv.csrrsi"


@irdl_op_definition
class CsrrciOp(CsrBitwiseImmOperation):
    """
    Reads the value of the CSR, zero-extends the value to XLEN bits, and writes
    it to integer register rd.  The value in the 'immediate' attribute is treated
    as a bit mask that specifies bit positions to be cleared in the CSR.
    Any bit that is high in rs1 will cause the corresponding bit to be cleared in the CSR,
    if that CSR bit is writable. Other bits in the CSR are unaffected (though CSRs might
    have side effects when written).

    If the 'immediate' attribute value is zero, then the instruction will not write
    to the CSR at all, and so shall not cause any of the side effects that might otherwise
    occur on a CSR write, such as raising illegal instruction exceptions on accesses to
    read-only CSRs.

    t = CSRs[csr]; CSRs[csr] = t &~zimm; x[rd] = t

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrci
    """

    name = "riscv.csrrci"


# endregion

# region RV32M/RV64M: 7 “M” Standard Extension for Integer Multiplication and Division

## Multiplication Operations


@irdl_op_definition
class MulOp(RdRsRsOperation):
    """
    Performs an XLEN-bit × XLEN-bit multiplication of signed rs1 by signed rs2
    and places the lower XLEN bits in the destination register.
    x[rd] = x[rs1] * x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#add
    """

    name = "riscv.mul"


class MulhOp(RdRsRsOperation):
    """
    Performs an XLEN-bit × XLEN-bit multiplication of signed rs1 by signed rs2
    and places the upper XLEN bits in the destination register.
    x[rd] = (x[rs1] s×s x[rs2]) >>s XLEN

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#mulh
    """

    name = "riscv.mulh"


class MulhsuOp(RdRsRsOperation):
    """
    Performs an XLEN-bit × XLEN-bit multiplication of signed rs1 by unsigned rs2
    and places the upper XLEN bits in the destination register.
    x[rd] = (x[rs1] s × x[rs2]) >>s XLEN

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#mulhsu
    """

    name = "riscv.mulhsu"


class MulhuOp(RdRsRsOperation):
    """
    Performs an XLEN-bit × XLEN-bit multiplication of unsigned rs1 by unsigned rs2
    and places the upper XLEN bits in the destination register.
    x[rd] = (x[rs1] u × x[rs2]) >>u XLEN

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#mulhu
    """

    name = "riscv.mulhu"


## Division Operations
class DivOp(RdRsRsOperation):
    """
    Perform an XLEN bits by XLEN bits signed integer division of rs1 by rs2,
    rounding towards zero.
    x[rd] = x[rs1] /s x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#div
    """

    name = "riscv.div"


class DivuOp(RdRsRsOperation):
    """
    Perform an XLEN bits by XLEN bits unsigned integer division of rs1 by rs2,
    rounding towards zero.
    x[rd] = x[rs1] /u x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#divu
    """

    name = "riscv.divu"


class RemOp(RdRsRsOperation):
    """
    Perform an XLEN bits by XLEN bits signed integer reminder of rs1 by rs2.
    x[rd] = x[rs1] %s x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#rem
    """

    name = "riscv.rem"


class RemuOp(RdRsRsOperation):
    """
    Perform an XLEN bits by XLEN bits unsigned integer reminder of rs1 by rs2.
    x[rd] = x[rs1] %u x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#remu
    """

    name = "riscv.remu"


# endregion

# region Assembler pseudo-instructions
# https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md


@irdl_op_definition
class LiOp(RdImmOperation):
    """
    Loads an immediate into rd.

    This is an assembler pseudo-instruction.

    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#load-immediate
    """

    name = "riscv.li"


@irdl_op_definition
class EcallOp(NullaryOperation):
    """
    The ECALL instruction is used to make a request to the supporting execution
    environment, which is usually an operating system.
    The ABI for the system will define how parameters for the environment
    request are passed, but usually these will be in defined locations in the
    integer register file.

    https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf
    """

    name = "riscv.ecall"


@irdl_op_definition
class EbreakOp(NullaryOperation):
    """
    The EBREAK instruction is used by debuggers to cause control to be
    transferred back to a debugging environment.

    https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf
    """

    name = "riscv.ebreak"


@irdl_op_definition
class WfiOp(NullaryOperation):
    """
    The Wait for Interrupt instruction (WFI) provides a hint to the
    implementation that the current hart can be stalled until an
    interrupt might need servicing.

    https://github.com/riscv/riscv-isa-manual/releases/download/Priv-v1.12/riscv-privileged-20211203.pdf
    """

    name = "riscv.wfi"


# endregion

# RISC-V SSA Helpers


@irdl_op_definition
class GetRegisterOp(IRDLOperation, RISCVOp):
    """
    This instruction allows us to create an SSAValue with for a given register name. This
    is useful for bridging the RISC-V convention that stores the result of function calls
    in `a0` and `a1` into SSA form.

    For example, to generate this assembly:
    ```
    jal my_func
    add a0 s0 a0
    ```

    One needs to do the following:

    ``` python
    rhs = riscv.GetRegisterOp(Registers.s0).res
    riscv.JalOp("my_func")
    lhs = riscv.GetRegisterOp(Registers.A0).res
    sum = riscv.AddOp(lhs, rhs, Registers.A0).rd
    ```
    """

    name = "riscv.get_register"
    res: Annotated[OpResult, RegisterType]

    def __init__(
        self,
        register_type: RegisterType | Register,
    ):
        if isinstance(register_type, Register):
            register_type = RegisterType(register_type)
        super().__init__(result_types=[register_type])


RISCV = Dialect(
    [
        AddiOp,
        SltiOp,
        SltiuOp,
        AndiOp,
        OriOp,
        XoriOp,
        SlliOp,
        SrliOp,
        SraiOp,
        LuiOp,
        AuipcOp,
        MVOp,
        AddOp,
        SltOp,
        SltuOp,
        AndOp,
        OrOp,
        XorOp,
        SllOp,
        SrlOp,
        SubOp,
        SraOp,
        NopOp,
        JalOp,
        JOp,
        JalrOp,
        ReturnOp,
        BeqOp,
        BneOp,
        BltOp,
        BgeOp,
        BltuOp,
        BgeuOp,
        LbOp,
        LbuOp,
        LhOp,
        LhuOp,
        LwOp,
        SbOp,
        ShOp,
        SwOp,
        CsrrwOp,
        CsrrsOp,
        CsrrcOp,
        CsrrwiOp,
        CsrrsiOp,
        CsrrciOp,
        MulOp,
        MulhOp,
        MulhsuOp,
        MulhuOp,
        DivOp,
        DivuOp,
        RemOp,
        RemuOp,
        LiOp,
        EcallOp,
        EbreakOp,
        WfiOp,
        GetRegisterOp,
    ],
    [
        RegisterType,
        LabelAttr,
    ],
)
