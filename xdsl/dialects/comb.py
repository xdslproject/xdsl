"""
The comb dialect provides a collection of operations that define a mid-level
compiler IR for combinational logic. It is designed to be easy to analyze and
transform, and be a flexible and extensible substrate that may be extended with
higher level dialects mixed into it.

[1] https://circt.llvm.org/docs/Dialects/Comb/
"""
from abc import ABC
from collections.abc import Sequence
from typing import Annotated

from xdsl.dialects.builtin import IntegerAttr, IntegerType, UnitAttr, i32, i64
from xdsl.ir import Attribute, Dialect, Operation, OpResult, SSAValue
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
from xdsl.utils.exceptions import VerifyException

ICMP_COMPARISON_OPERATIONS = [
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
]


class BinCombOperation(IRDLOperation, ABC):
    """
    A binary comb operation. It has two operands and one
    result, all of the same integer type.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)
    """
    "All operations are defined in the expected way for 2-state (binary) logic. However, comb is
    used for operations which have extended truth table for non-2-state logic for various target
    languages. The two_state variable describes if we are using 2-state (binary) logic or not."
    """
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
        printer.print(self.result.type)


class VariadicCombOperation(IRDLOperation, ABC):
    """
    A variadic comb operation. It has a variadic number of operands, and a single
    result, all of the same integer type.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    inputs: VarOperand = var_operand_def(T)
    result: OpResult = result_def(T)
    """
    "All operations are defined in the expected way for 2-state (binary) logic. However, comb is
    used for operations which have extended truth table for non-2-state logic for various target
    languages. The two_state variable describes if we are using 2-state (binary) logic or not."
    """
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
        inputs = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        inputs = parser.resolve_operands(
            inputs, len(inputs) * [result_type], parser.pos
        )
        return cls.create(operands=inputs, result_types=[result_type])

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_list(self.inputs, printer.print_ssa_value)
        printer.print(" : ")
        printer.print(self.result.type)


@irdl_op_definition
class AddOp(VariadicCombOperation):
    """Addition"""

    name = "comb.add"


@irdl_op_definition
class MulOp(VariadicCombOperation):
    """Multiplication"""

    name = "comb.mul"


@irdl_op_definition
class DivUOp(BinCombOperation):
    """Unsigned division"""

    name = "comb.divu"


@irdl_op_definition
class DivSOp(BinCombOperation):
    """Signed division"""

    name = "comb.divs"


@irdl_op_definition
class ModUOp(BinCombOperation):
    """Unsigned remainder"""

    name = "comb.modu"


@irdl_op_definition
class ModSOp(BinCombOperation):
    """Signed remainder"""

    name = "comb.mods"


@irdl_op_definition
class ShlOp(BinCombOperation):
    """Left shift"""

    name = "comb.shl"


@irdl_op_definition
class ShrUOp(BinCombOperation):
    """Unsigned right shift"""

    name = "comb.shru"


@irdl_op_definition
class ShrSOp(BinCombOperation):
    """Signed right shift"""

    name = "comb.shrs"


@irdl_op_definition
class SubOp(BinCombOperation):
    """Subtraction"""

    name = "comb.sub"


@irdl_op_definition
class AndOp(VariadicCombOperation):
    """Bitwise and"""

    name = "comb.and"


@irdl_op_definition
class OrOp(VariadicCombOperation):
    """Bitwise or"""

    name = "comb.or"


@irdl_op_definition
class XorOp(VariadicCombOperation):
    """Bitwise xor"""

    name = "comb.xor"


@irdl_op_definition
class ICmpOp(IRDLOperation, ABC):
    """Integer comparison: A generic comparison operation, operation definitions inherit this class.

    The first argument to these comparison operations is the type of comparison
    being performed, the following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   signed less than (mnemonic: `"slt"`; integer value: `2`)
    -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
    """

    name = "comb.icmp"

    T = Annotated[IntegerType, ConstraintVar("T")]

    predicate: IntegerAttr[IntegerType] = attr_def(
        IntegerAttr[Annotated[IntegerType, i64]]
    )
    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(IntegerType(1))

    two_state: UnitAttr = attr_def(UnitAttr)

    @staticmethod
    def _get_comparison_predicate(
        mnemonic: str, comparison_operations: dict[str, int]
    ) -> int:
        if mnemonic in comparison_operations:
            return comparison_operations[mnemonic]
        else:
            raise VerifyException(f"Unknown comparison mnemonic: {mnemonic}")

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        arg: int | str | IntegerAttr[IntegerType],
    ):
        operand1 = SSAValue.get(operand1)
        operand2 = SSAValue.get(operand2)

        if isinstance(arg, str):
            cmpi_comparison_operations = {
                "eq": 0,
                "ne": 1,
                "slt": 2,
                "sle": 3,
                "sgt": 4,
                "sge": 5,
                "ult": 6,
                "ule": 7,
                "ugt": 8,
                "uge": 9,
            }
            arg = ICmpOp._get_comparison_predicate(arg, cmpi_comparison_operations)
        if not isinstance(arg, IntegerAttr):
            arg = IntegerAttr.from_int_and_width(arg, 64)
        return super().__init__(
            operands=[operand1, operand2],
            result_types=[IntegerType(1)],
            attributes={"predicate": arg},
        )

    @classmethod
    def parse(cls, parser: Parser):
        arg = parser.parse_identifier()
        parser.parse_punctuation(",")
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        (operand1, operand2) = parser.resolve_operands(
            [operand1, operand2], 2 * [input_type], parser.pos
        )

        return cls(operand1, operand2, arg)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_string(ICMP_COMPARISON_OPERATIONS[self.predicate.value.data])
        printer.print(", ")
        printer.print_operand(self.lhs)
        printer.print(", ")
        printer.print_operand(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.lhs.type)


@irdl_op_definition
class ParityOp(IRDLOperation):
    """Parity"""

    name = "comb.parity"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType(1))

    two_state: UnitAttr | None = opt_attr_def(UnitAttr)

    def __init__(
        self, operand: Operation | SSAValue, two_state: UnitAttr | None = None
    ):
        operand = SSAValue.get(operand)
        return super().__init__(
            attributes={"two_state": two_state},
            operands=[operand],
            result_types=[operand.type],
        )

    @classmethod
    def parse(cls, parser: Parser):
        op = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        op = parser.resolve_operand(op, result_type)
        return cls(op)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print(" : ")
        printer.print(self.result.type)


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

    def __init__(
        self, operand: Operation | SSAValue, low_bit: IntegerAttr[IntegerType]
    ):
        operand = SSAValue.get(operand)
        return super().__init__(
            attributes={"low_bit": low_bit},
            operands=[operand],
            result_types=[operand.type],
        )

    @classmethod
    def parse(cls, parser: Parser):
        op = parser.parse_unresolved_operand()
        parser.parse_keyword("from")
        bit = parser.parse_integer()
        parser.parse_punctuation(":")
        result_type = parser.parse_function_type()
        (op,) = parser.resolve_operands([op], result_type.inputs.data, parser.pos)
        return cls(op, IntegerAttr(bit, 32))

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print(" : ")
        printer.print(self.result.type)


@irdl_op_definition
class ConcatOp(IRDLOperation):
    """
    Concatenate a variadic list of operands together.
    """

    name = "comb.concat"

    inputs: VarOperand = var_operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        return super().__init__(operands=[op], result_types=[target_type])

    @classmethod
    def parse(cls, parser: Parser):
        inputs = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        inputs = parser.resolve_operands(
            inputs, len(inputs) * [result_type], parser.pos
        )
        return cls.create(operands=inputs, result_types=[result_type])

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_list(self.inputs, printer.print_ssa_value)
        printer.print(" : ")
        printer.print(self.result.type)


@irdl_op_definition
class ReplicateOp(IRDLOperation):
    """
    Concatenate the operand a constant number of times.
    """

    name = "comb.replicate"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        return super().__init__(operands=[op], result_types=[target_type])

    @classmethod
    def parse(cls, parser: Parser):
        op = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_function_type()
        (op,) = parser.resolve_operands([op], [result_type.inputs.data[0]], parser.pos)
        return cls.create(operands=[op], result_types=result_type.outputs.data)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print(" : ")
        printer.print(self.result.type)


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

    def __init__(
        self,
        condition: Operation | SSAValue,
        true_val: Operation | SSAValue,
        false_val: Operation | SSAValue,
    ):
        operand2 = SSAValue.get(true_val)
        return super().__init__(
            operands=[condition, true_val, false_val], result_types=[operand2.type]
        )

    @classmethod
    def parse(cls, parser: Parser):
        condition = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        true_val = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        false_val = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        (condition, true_val, false_val) = parser.resolve_operands(
            [condition, true_val, false_val],
            [IntegerType(1), result_type, result_type],
            parser.pos,
        )
        return cls(condition, true_val, false_val)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.cond)
        printer.print(", ")
        printer.print_operand(self.true_value)
        printer.print(", ")
        printer.print_operand(self.false_value)
        printer.print(" : ")
        printer.print_attribute(self.result.type)


Comb = Dialect(
    "comb",
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
    ],
)
