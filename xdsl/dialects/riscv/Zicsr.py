from abc import ABC

from xdsl.dialects.builtin import AnyIntegerAttr, StringAttr, UnitAttr
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.ir.core import Dialect
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException

from .base import IntegerRegister, IntegerRegisterType
from .core import AssemblyInstructionArgType, RISCVInstruction


class CsrOperation(IRDLOperation, RISCVInstruction, ABC):
    rd: OpResult = result_def(IntegerRegisterType)
    csr: AnyIntegerAttr = attr_def(AnyIntegerAttr)


class CsrReadWriteOperation(CsrOperation):
    """
    A base class for RISC-V operations performing a swap to/from a CSR.
    The 'writeonly' attribute controls the actual behaviour of the operation:
    * when True, the operation writes the rs value to the CSR but never reads it and
      in this case rd *must* be allocated to x0
    * when False, a proper atomic swap is performed and the previous CSR value is
      returned in rd
    """

    rs1: Operand = operand_def(IntegerRegisterType)
    writeonly: UnitAttr | None = opt_attr_def(UnitAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        csr: AnyIntegerAttr,
        *,
        writeonly: bool = False,
        rd: IntegerRegisterType | IntegerRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = IntegerRegisterType(IntegerRegister())
        elif isinstance(rd, IntegerRegister):
            rd = IntegerRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs1],
            attributes={
                "csr": csr,
                "writeonly": UnitAttr() if writeonly else None,
                "comment": comment,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if not self.writeonly:
            return
        if not isinstance(self.rd.typ, IntegerRegisterType):
            return
        if self.rd.typ.data.name is not None and self.rd.typ.data.name != "zero":
            raise VerifyException(
                "When in 'writeonly' mode, destination must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rd.typ.data.name}'"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.csr, self.rs1


class CsrBitwiseOperation(CsrOperation):
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

    rs1: Operand = operand_def(IntegerRegisterType)
    readonly: UnitAttr | None = opt_attr_def(UnitAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        csr: AnyIntegerAttr,
        *,
        readonly: bool = False,
        rd: IntegerRegisterType | IntegerRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = IntegerRegisterType(IntegerRegister())
        elif isinstance(rd, IntegerRegister):
            rd = IntegerRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs1],
            attributes={
                "csr": csr,
                "readonly": UnitAttr() if readonly else None,
                "comment": comment,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if not self.readonly:
            return
        if not isinstance(self.rs1.typ, IntegerRegisterType):
            return
        if self.rs1.typ.data.name is not None and self.rs1.typ.data.name != "zero":
            raise VerifyException(
                "When in 'readonly' mode, source must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rs1.typ.data.name}'"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.csr, self.rs1


class CsrReadWriteImmOperation(CsrOperation):
    """
    A base class for RISC-V operations performing a write immediate to/read from a CSR.
    The 'writeonly' attribute controls the actual behaviour of the operation:
    * when True, the operation writes the rs value to the CSR but never reads it and
      in this case rd *must* be allocated to x0
    * when False, a proper atomic swap is performed and the previous CSR value is
      returned in rd
    """

    writeonly: UnitAttr | None = opt_attr_def(UnitAttr)
    immediate: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

    def __init__(
        self,
        csr: AnyIntegerAttr,
        immediate: AnyIntegerAttr,
        *,
        writeonly: bool = False,
        rd: IntegerRegisterType | IntegerRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = IntegerRegisterType(IntegerRegister())
        elif isinstance(rd, IntegerRegister):
            rd = IntegerRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            attributes={
                "csr": csr,
                "immediate": immediate,
                "writeonly": UnitAttr() if writeonly else None,
                "comment": comment,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if self.writeonly is None:
            return
        if not isinstance(self.rd.typ, IntegerRegisterType):
            return
        if self.rd.typ.data.name is not None and self.rd.typ.data.name != "zero":
            raise VerifyException(
                "When in 'writeonly' mode, destination must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rd.typ.data.name}'"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.csr, self.immediate


class CsrBitwiseImmOperation(CsrOperation):
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

    immediate: AnyIntegerAttr = attr_def(AnyIntegerAttr)

    def __init__(
        self,
        csr: AnyIntegerAttr,
        immediate: AnyIntegerAttr,
        *,
        rd: IntegerRegisterType | IntegerRegister | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = IntegerRegisterType(IntegerRegister())
        elif isinstance(rd, IntegerRegister):
            rd = IntegerRegisterType(rd)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            attributes={
                "csr": csr,
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
        return self.rd, self.csr, self.immediate


# region RV32/RV64 Zicsr Standard Extension


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

RISCV_ZICSR = Dialect(
    [
        CsrrwOp,
        CsrrsOp,
        CsrrcOp,
        CsrrwiOp,
        CsrrsiOp,
        CsrrciOp,
    ],
    [],
)
