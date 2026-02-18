"""
The comb dialect provides a collection of operations that define a mid-level
compiler IR for combinational logic. It is designed to be easy to analyze and
transform, and be a flexible and extensible substrate that may be extended with
higher level dialects mixed into it.

Up to date as of CIRCT commit `2e23cda6c2cbedb118b92fab755f1e36d80b13f5`.

See external [documentation](https://circt.llvm.org/docs/Dialects/Comb/).
"""

from abc import ABC
from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import (
    I32,
    I64,
    IntegerAttr,
    IntegerType,
    UnitAttr,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    VarConstraint,
    attr_def,
    base,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

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

    T: ClassVar = VarConstraint("T", base(IntegerType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)
    """
    "All operations are defined in the expected way for 2-state (binary) logic. However, comb is
    used for operations which have extended truth table for non-2-state logic for various target
    languages. The two_state variable describes if we are using 2-state (binary) logic or not."
    """
    two_state = opt_attr_def(UnitAttr)

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
        printer.print_string(" ")
        printer.print_ssa_value(self.lhs)
        printer.print_string(", ")
        printer.print_ssa_value(self.rhs)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)


class VariadicCombOperation(IRDLOperation, ABC):
    """
    A variadic comb operation. It has a variadic number of operands, and a single
    result, all of the same integer type.
    """

    T: ClassVar = VarConstraint("T", base(IntegerType))

    inputs = var_operand_def(T)
    result = result_def(T)
    """
    "All operations are defined in the expected way for 2-state (binary) logic. However, comb is
    used for operations which have extended truth table for non-2-state logic for various target
    languages. The two_state variable describes if we are using 2-state (binary) logic or not."
    """
    two_state = opt_attr_def(UnitAttr)

    def __init__(
        self,
        input_list: Sequence[Operation | SSAValue],
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            if len(input_list) == 0:
                raise ValueError("cannot infer type from zero inputs")
            result_type = SSAValue.get(input_list[0]).type
        super().__init__(operands=[input_list], result_types=[result_type])

    def verify_(self) -> None:
        if len(self.inputs) == 0:
            raise VerifyException("op expected 1 or more operands, but found 0")

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
        printer.print_string(" ")
        printer.print_list(self.inputs, printer.print_ssa_value)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)


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

    T: ClassVar = VarConstraint("T", base(IntegerType))

    predicate = attr_def(IntegerAttr[I64])
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(IntegerType(1))

    two_state = opt_attr_def(UnitAttr, attr_name="twoState")

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
        has_two_state_semantics: bool = False,
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
            arg = IntegerAttr(arg, 64)

        attrs: dict[str, Attribute] = {"predicate": arg}
        if has_two_state_semantics:
            attrs["twoState"] = UnitAttr()

        return super().__init__(
            operands=[operand1, operand2],
            result_types=[IntegerType(1)],
            attributes=attrs,
        )

    @classmethod
    def parse(cls, parser: Parser):
        has_two_state_semantics = parser.parse_optional_keyword("bin") is not None
        arg = parser.parse_identifier()
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        (operand1, operand2) = parser.resolve_operands(
            [operand1, operand2], 2 * [input_type], parser.pos
        )

        return cls(operand1, operand2, arg, has_two_state_semantics)

    def print(self, printer: Printer):
        printer.print_string(" ")
        if self.two_state is not None:
            printer.print_string("bin ")
        printer.print_string(ICMP_COMPARISON_OPERATIONS[self.predicate.value.data])
        printer.print_string(" ")
        printer.print_operand(self.lhs)
        printer.print_string(", ")
        printer.print_operand(self.rhs)
        printer.print_string(" : ")
        printer.print_attribute(self.lhs.type)


@irdl_op_definition
class ParityOp(IRDLOperation):
    """Parity"""

    name = "comb.parity"

    input = operand_def(IntegerType)
    result = result_def(IntegerType(1))

    two_state = opt_attr_def(UnitAttr, attr_name="twoState")

    def __init__(
        self, operand: Operation | SSAValue, two_state: UnitAttr | None = None
    ):
        operand = SSAValue.get(operand)
        return super().__init__(
            attributes={"twoState": two_state},
            operands=[operand],
            result_types=[operand.type],
        )

    @classmethod
    def parse(cls, parser: Parser):
        two_state = None
        if parser.parse_optional_keyword("bin") is not None:
            two_state = UnitAttr()
        op = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        op = parser.resolve_operand(op, result_type)
        return cls(op, two_state)

    def print(self, printer: Printer):
        printer.print_string(" ")
        if self.two_state is not None:
            printer.print_string("bin ")
        printer.print_ssa_value(self.input)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """
    Extract a range of bits into a smaller value, low_bit
    specifies the lowest bit included. Result is the size
    of the value to extract.

    |-----------------|   input
           l              low bit
           <-------->     result
    """

    name = "comb.extract"

    input = operand_def(IntegerType)
    low_bit = attr_def(IntegerAttr[I32], attr_name="lowBit")
    result = result_def(IntegerType)

    def __init__(
        self,
        operand: Operation | SSAValue,
        low_bit: IntegerAttr[IntegerType],
        result_type: IntegerType,
    ):
        operand = SSAValue.get(operand)
        return super().__init__(
            attributes={"lowBit": low_bit},
            operands=[operand],
            result_types=[result_type],
        )

    def verify_(self) -> None:
        assert isa(self.input.type, IntegerType)
        assert isinstance(self.result.type, IntegerType)
        if (
            self.low_bit.value.data + self.result.type.width.data
            > self.input.type.width.data + 1
        ):
            raise VerifyException(
                f"output width {self.result.type.width.data} is "
                f"too large for input of width "
                f"{self.input.type.width.data} (included low bit "
                f"is at {self.low_bit.value.data})"
            )

    @classmethod
    def parse(cls, parser: Parser):
        op = parser.parse_unresolved_operand()
        parser.parse_keyword("from")
        bit = parser.parse_integer()
        parser.parse_punctuation(":")
        result_type = parser.parse_function_type()
        if len(result_type.inputs.data) != 1 or len(result_type.outputs.data) != 1:
            parser.raise_error(
                "expected exactly one input and exactly one output types"
            )
        if not isa(result_type.outputs.data[0], IntegerType):
            parser.raise_error(
                f"expected output to be an integer type, got '{result_type.outputs.data[0]}'"
            )
        (op,) = parser.resolve_operands([op], result_type.inputs.data, parser.pos)
        return cls(op, IntegerAttr(bit, 32), result_type.outputs.data[0])

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.input)
        printer.print_string(f" from {self.low_bit.value.data} : ")
        printer.print_function_type([self.input.type], [self.result.type])


def _get_sum_of_int_width(int_types: Sequence[Attribute]) -> int | None:
    """
    Gets the sum of the width of the provided integer types. Returns None
    if one of the provided attributes is not an integer type.
    """
    sum_of_width = 0
    for typ in int_types:
        if not isa(typ, IntegerType):
            return None
        sum_of_width += typ.width.data
    return sum_of_width


@irdl_op_definition
class ConcatOp(IRDLOperation):
    """
    Concatenate a variadic list of operands together.
    """

    name = "comb.concat"

    inputs = var_operand_def(IntegerType)
    result = result_def(IntegerType)

    def __init__(self, ops: Sequence[SSAValue | Operation], target_type: IntegerType):
        return super().__init__(operands=[ops], result_types=[target_type])

    @staticmethod
    def from_int_values(inputs: Sequence[SSAValue]) -> "ConcatOp | None":
        """
        Concatenates the provided values, in order. Returns None if the provided
        values are not integers.
        """
        sum_of_width = _get_sum_of_int_width([inp.type for inp in inputs])
        if sum_of_width is None:
            return None
        return ConcatOp(inputs, IntegerType(sum_of_width))

    def verify_(self) -> None:
        sum_of_width = _get_sum_of_int_width(self.inputs.types)
        assert sum_of_width is not None
        assert isinstance(self.result.type, IntegerType)
        if sum_of_width != self.result.type.width.data:
            raise VerifyException(
                f"Sum of integer width ({sum_of_width}) "
                f"is different from result "
                f"width ({self.result.type.width.data})"
            )

    @classmethod
    def parse(cls, parser: Parser):
        inputs = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )
        parser.parse_punctuation(":")
        input_types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_type
        )
        sum_of_width = _get_sum_of_int_width(input_types)
        if sum_of_width is None:
            parser.raise_error("expected only integer types as input")
        inputs = parser.resolve_operands(inputs, input_types, parser.pos)
        return cls.create(operands=inputs, result_types=[IntegerType(sum_of_width)])

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_list(self.inputs, printer.print_ssa_value)
        printer.print_string(" : ")
        printer.print_list(self.inputs.types, printer.print_attribute)


@irdl_op_definition
class ReplicateOp(IRDLOperation):
    """
    Concatenate the operand a constant number of times.
    """

    name = "comb.replicate"

    input = operand_def(IntegerType)
    result = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        return super().__init__(operands=[op], result_types=[target_type])

    @classmethod
    def parse(cls, parser: Parser):
        op = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        fun_type = parser.parse_function_type()
        operands = parser.resolve_operands([op], fun_type.inputs.data, parser.pos)
        return cls.create(operands=operands, result_types=fun_type.outputs.data)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.input)
        printer.print_string(" : ")
        printer.print_function_type((self.input.type,), (self.result.type,))


@irdl_op_definition
class MuxOp(IRDLOperation):
    """
    Select between two values based on a condition.
    """

    name = "comb.mux"

    T: ClassVar = VarConstraint("T", base(TypeAttribute))

    cond = operand_def(IntegerType(1))
    true_value = operand_def(T)
    false_value = operand_def(T)
    result = result_def(T)

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
        printer.print_string(" ")
        printer.print_operand(self.cond)
        printer.print_string(", ")
        printer.print_operand(self.true_value)
        printer.print_string(", ")
        printer.print_operand(self.false_value)
        printer.print_string(" : ")
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
