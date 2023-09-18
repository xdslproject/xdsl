from collections.abc import Sequence
from typing import Annotated

from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, UnitAttr, i32
from xdsl.ir import Dialect, Operation, OpResult, SSAValue
from xdsl.ir.core import Attribute
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    Operand,
    VarOperand,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


class BinCombOp(IRDLOperation):
    """
    A binary comb operation. It has two operands and one
    result, all of the same integer type.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)

    two_state: UnitAttr | None = opt_attr_def(UnitAttr)

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        super().__init__(operands=[operand1, operand2], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser):
        lhs = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        rhs = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        (lhs, rhs) = parser.resolve_operands([lhs, rhs], 2 * [result_type], parser.pos)
        return cls(lhs, rhs, result_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.result.type)

    def __hash__(self) -> int:
        return id(self)


class VariadicCombOp(IRDLOperation):
    """
    A variadic comb operation. It has a variadic number of operands, and a single
    result, all of the same type.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    inputs: VarOperand = var_operand_def(T)
    result: OpResult = result_def(T)

    two_state: UnitAttr | None = opt_attr_def(UnitAttr)

    def __init__(
        self,
        input_list: Sequence[Operation | SSAValue],
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(input_list[0]).type
        super().__init__(operands=input_list, result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser):
        inputs = parser.parse_op_args_list()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        inputs = parser.resolve_operands(
            inputs, len(inputs) * [result_type], parser.pos
        )
        return cls(inputs, result_type)

    def print(self, printer: Printer):
        printer.print(" ")
        for item in self.inputs:
            printer.print_ssa_value(item)
            printer.print(", ")
        printer.print(" : ")
        printer.print_attribute(self.result.type)

    def __hash__(self) -> int:
        return id(self)


@irdl_op_definition
class AddOp(VariadicCombOp):
    """Addition"""

    name = "comb.add"


@irdl_op_definition
class MulOp(VariadicCombOp):
    """Multiplication"""

    name = "comb.mul"


@irdl_op_definition
class DivUOp(BinCombOp):
    """Unsigned division"""

    name = "comb.divu"


@irdl_op_definition
class DivSOp(BinCombOp):
    """Signed division"""

    name = "comb.divs"


@irdl_op_definition
class ModUOp(BinCombOp):
    """Unsigned remainder"""

    name = "comb.modu"


@irdl_op_definition
class ModSOp(BinCombOp):
    """Signed remainder"""

    name = "comb.mods"


@irdl_op_definition
class ShlOp(BinCombOp):
    """Left shift"""

    name = "comb.shl"


@irdl_op_definition
class ShrUOp(BinCombOp):
    """Unsigned right shift"""

    name = "comb.shru"


@irdl_op_definition
class ShrSOp(BinCombOp):
    """Signed right shift"""

    name = "comb.shrs"


@irdl_op_definition
class SubOp(BinCombOp):
    """Subtraction"""

    name = "comb.sub"


@irdl_op_definition
class AndOp(VariadicCombOp):
    """Bitwise and"""

    name = "comb.and"


@irdl_op_definition
class OrOp(VariadicCombOp):
    """Bitwise or"""

    name = "comb.or"


@irdl_op_definition
class XorOp(VariadicCombOp):
    """Bitwise xor"""

    name = "comb.xor"


@irdl_op_definition
class ICmpOp(IRDLOperation):
    """Integer comparison"""

    name = "comb.icmp"

    T = Annotated[IntegerType, ConstraintVar("T")]

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(IntegerType(1))

    predicate: IntegerAttr[IndexType] = attr_def(IntegerAttr[IndexType])
    two_state: UnitAttr = attr_def(UnitAttr)


@irdl_op_definition
class ParityOp(IRDLOperation):
    """Parity"""

    name = "comb.parity"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType(1))

    two_state: UnitAttr | None = opt_attr_def(UnitAttr)


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """
    Extract a range of bits into a smaller value, low_bit
    specifies the lowest bit included.
    """

    name = "comb.extract"

    input: Operand = operand_def(IntegerType)
    low_bit: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    result: OpResult = result_def(IntegerType)


@irdl_op_definition
class ConcatOp(IRDLOperation):
    """
    Concatenate a variadic list of operands together.
    """

    name = "comb.concat"

    inputs: VarOperand = var_operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)


@irdl_op_definition
class ReplicateOp(IRDLOperation):
    """
    Concatenate the operand a constant number of times.
    """

    name = "comb.replicate"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)


@irdl_op_definition
class MuxOp(IRDLOperation):
    """
    Select between two values based on a condition.
    """

    name = "comb.mux"

    T = Annotated[IntegerType, ConstraintVar("T")]

    cond: Operand = operand_def(IntegerType(1))
    true_value: Operand = operand_def(T)
    false_value: Operand = operand_def(T)
    result: OpResult = result_def(T)


Comb = Dialect(
    [
        AddOp,
        MulOp,
        DivUOp,
        DivSOp,
        ModUOp,
        ModSOp,
        ShlOp,
        ShrUOp,
        ShrSOp,
        SubOp,
        AndOp,
        OrOp,
        XorOp,
        ICmpOp,
        ParityOp,
        ExtractOp,
        ConcatOp,
        ReplicateOp,
        MuxOp,
    ]
)
