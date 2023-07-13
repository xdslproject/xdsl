from __future__ import annotations

from io import StringIO

import pytest
from conftest import assert_print_op

from xdsl.dialects.arith import Addi, Arith, Constant
from xdsl.dialects.builtin import Builtin, IntAttr, IntegerType, UnitAttr, i32
from xdsl.dialects.func import Func
from xdsl.dialects.test import Test, TestOp
from xdsl.ir import (
    Attribute,
    Block,
    MLContext,
    Operation,
    OpResult,
    ParametrizedAttribute,
    Region,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    ParameterDef,
    VarOperand,
    VarOpResult,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.utils.diagnostic import Diagnostic
from xdsl.utils.exceptions import ParseError


def test_simple_forgotten_op():
    """Test that the parsing of an undefined operand gives it a name."""
    ctx = MLContext()
    ctx.register_dialect(Arith)

    lit = Constant.from_int_and_width(42, 32)
    add = Addi(lit, lit)

    add.verify()

    expected = """%0 = "arith.addi"(%1, %1) : (i32, i32) -> i32"""

    assert_print_op(add, expected, None)


def test_print_op_location():
    """Test that an op can be printed with its location."""
    ctx = MLContext()
    ctx.register_dialect(Test)

    add = TestOp(operands=[[]], result_types=[[i32]], regions=[[]])

    add.verify()

    expected = """%0 = "test.op"() : () -> i32 loc(unknown)"""

    assert_print_op(add, expected, None, print_debuginfo=True)


@irdl_op_definition
class UnitAttrOp(IRDLOperation):
    name = "unit_attr_op"

    parallelize: UnitAttr | None = opt_attr_def(UnitAttr)


def test_unit_attr():
    """Test that a UnitAttr can be defined and printed"""

    expected = """
"unit_attr_op"() {"parallelize"} : () -> ()
"""

    unit_op = UnitAttrOp.build(attributes={"parallelize": UnitAttr([])})

    assert_print_op(unit_op, expected, None)


def test_added_unit_attr():
    """Test that a UnitAttr can be added to an op, even if its not defined as a field."""

    expected = """
"unit_attr_op"() {"parallelize", "vectorize"} : () -> ()
"""
    unitop = UnitAttrOp.build(
        attributes={"parallelize": UnitAttr([]), "vectorize": UnitAttr([])}
    )

    assert_print_op(unitop, expected, None)


#  ____  _                             _   _
# |  _ \(_) __ _  __ _ _ __   ___  ___| |_(_) ___
# | | | | |/ _` |/ _` | '_ \ / _ \/ __| __| |/ __|
# | |_| | | (_| | (_| | | | | (_) \__ \ |_| | (__
# |____/|_|\__,_|\__, |_| |_|\___/|___/\__|_|\___|
#                |___/
#


def test_op_message():
    """Test that an operation message can be printed."""
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  ^^^^^^^^^^^^^^^^^^^^^
  | Test message
  ---------------------
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()
"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    first_op = module.ops.first
    assert first_op is not None
    diagnostic.add_message(first_op, "Test message")

    assert_print_op(module, expected, diagnostic)


def test_two_different_op_messages():
    """Test that an operation message can be printed."""
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  ^^^^^^^^^^^^^^^^^^^^^
  | Test message 1
  ---------------------
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
  ^^^^^^^^^^^^^^^^^
  | Test message 2
  -----------------
}) : () -> ()"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    first_op, second_op = list(module.ops)
    diagnostic.add_message(first_op, "Test message 1")
    diagnostic.add_message(second_op, "Test message 2")

    assert_print_op(module, expected, diagnostic)


def test_two_same_op_messages():
    """Test that an operation message can be printed."""
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  ^^^^^^^^^^^^^^^^^^^^^
  | Test message 1
  ---------------------
  ^^^^^^^^^^^^^^^^^^^^^
  | Test message 2
  ---------------------
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    first_op, _second_op = list(module.ops)

    diagnostic.add_message(first_op, "Test message 1")
    diagnostic.add_message(first_op, "Test message 2")

    assert_print_op(module, expected, diagnostic)


def test_op_message_with_region():
    """Test that an operation message can be printed on an operation with a region."""
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
^^^^^^^^^^^^^^^^
| Test
----------------
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    diagnostic = Diagnostic()
    diagnostic.add_message(module, "Test")

    assert_print_op(module, expected, diagnostic)


def test_op_message_with_region_and_overflow():
    """
    Test that an operation message can be printed on an operation with a region,
    where the message is bigger than the operation.
    """
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
^^^^^^^^^^^^^^^^---
| Test long message
-------------------
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    diagnostic = Diagnostic()
    diagnostic.add_message(module, "Test long message")
    assert_print_op(module, expected, diagnostic)


def test_diagnostic():
    """
    Test that an operation message can be printed on an operation with a region,
    where the message is bigger than the operation.
    """
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    diag = Diagnostic()
    diag.add_message(module, "Test")
    try:
        diag.raise_exception("test message", module)
    except Exception as e:
        assert str(e)


#  ____ ____    _    _   _
# / ___/ ___|  / \  | \ | | __ _ _ __ ___   ___
# \___ \___ \ / _ \ |  \| |/ _` | '_ ` _ \ / _ \
#  ___) |__) / ___ \| |\  | (_| | | | | | |  __/
# |____/____/_/   \_\_| \_|\__,_|_| |_| |_|\___|
#


def test_print_custom_name():
    """
    Test that an SSAValue, that is a name and not a number, reserves that name
    """
    prog = """\
"builtin.module"() ({
  %i = arith.constant 42 : i32
  %213 = "arith.addi"(%i, %i) : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %i = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %0 = "arith.addi"(%i, %i) : (i32, i32) -> i32
}) : () -> ()
"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None)


def test_print_custom_block_arg_name():
    block = Block(arg_types=[i32, i32])
    block.args[0].name_hint = "test"
    block.args[1].name_hint = "test"

    io = StringIO()
    p = Printer(stream=io)
    p.print_block(block)
    assert io.getvalue() == """\n^0(%test : i32, %test_1 : i32):"""


def test_print_block_argument():
    """Print a block argument."""
    block = Block(arg_types=[i32, i32])

    io = StringIO()
    p = Printer(stream=io)
    p.print_block_argument(block.args[0])
    p.print(", ")
    p.print_block_argument(block.args[1], print_type=False)
    assert io.getvalue() == """%0 : i32, %1"""


def test_print_block_argument_location():
    """Print a block argument with location."""
    block = Block(arg_types=[i32, i32])

    io = StringIO()
    p = Printer(stream=io, print_debuginfo=True)
    p.print_block_argument(block.args[0])
    p.print(", ")
    p.print_block_argument(block.args[1])
    assert io.getvalue() == """%0 : i32 loc(unknown), %1 : i32 loc(unknown)"""


def test_print_block():
    """Print a block."""
    block = Block(arg_types=[i32, i32])
    block.add_op(TestOp(operands=(block.args[1],), result_types=[[]], regions=[[]]))

    # Print block arguments inside the block
    io = StringIO()
    p = Printer(stream=io)
    p.print_block(block)
    assert (
        io.getvalue() == """\n^0(%0 : i32, %1 : i32):\n  "test.op"(%1) : (i32) -> ()"""
    )


def test_print_block_without_arguments():
    """Print a block and its arguments separately."""
    block = Block(arg_types=[i32, i32])
    block.add_op(TestOp(operands=(block.args[1],), result_types=[[]], regions=[[]]))

    # Print block arguments separately from the block
    io = StringIO()
    p = Printer(stream=io)
    p.print_block_argument(block.args[0])
    p.print(", ")
    p.print_block_argument(block.args[1])
    p.print_block(block, print_block_args=False)
    assert io.getvalue() == """%0 : i32, %1 : i32\n  "test.op"(%1) : (i32) -> ()"""


def test_print_region():
    """Print a region."""
    block = Block(arg_types=[i32, i32])
    block.add_op(TestOp(operands=(block.args[1],), result_types=[[]], regions=[[]]))
    region = Region(block)

    io = StringIO()
    p = Printer(stream=io)
    p.print_region(region)
    assert (
        io.getvalue()
        == """{\n^0(%0 : i32, %1 : i32):\n  "test.op"(%1) : (i32) -> ()\n}"""
    )


def test_print_region_without_arguments():
    """Print a region and its arguments separately."""
    block = Block(arg_types=[i32, i32])
    block.add_op(TestOp(operands=(block.args[1],), result_types=[[]], regions=[[]]))
    region = Region(block)

    io = StringIO()
    p = Printer(stream=io)
    p.print_block_argument(block.args[0])
    p.print(", ")
    p.print_block_argument(block.args[1])
    p.print(" ")
    p.print_region(region, print_entry_block_args=False)
    assert io.getvalue() == """%0 : i32, %1 : i32 {\n  "test.op"(%1) : (i32) -> ()\n}"""


def test_print_region_empty_block():
    """
    Print a region with an empty block, and specify that
    empty entry blocks shouldn't be printed.
    """
    block = Block()
    region = Region(block)

    io = StringIO()
    p = Printer(stream=io)
    p.print_region(region, print_empty_block=False)
    assert io.getvalue() == """{\n}"""


def test_print_region_empty_block_with_args():
    """
    Print a region with an empty block and arguments, and specify that
    empty entry blocks shouldn't be printed.
    """
    block = Block(arg_types=[i32, i32])
    region = Region(block)

    io = StringIO()
    p = Printer(stream=io)
    p.print_region(region, print_empty_block=False)
    assert io.getvalue() == """{\n^0(%0 : i32, %1 : i32):\n}"""


#   ____          _                  _____                          _
#  / ___|   _ ___| |_ ___  _ __ ___ |  ___|__  _ __ _ __ ___   __ _| |_
# | |  | | | / __| __/ _ \| '_ ` _ \| |_ / _ \| '__| '_ ` _ \ / _` | __|
# | |__| |_| \__ \ || (_) | | | | | |  _| (_) | |  | | | | | | (_| | |_
#  \____\__,_|___/\__\___/|_| |_| |_|_|  \___/|_|  |_| |_| |_|\__,_|\__|
#


@irdl_op_definition
class PlusCustomFormatOp(IRDLOperation):
    name = "test.add"
    lhs: Operand = operand_def(IntegerType)
    rhs: Operand = operand_def(IntegerType)
    res: OpResult = result_def(IntegerType)

    @classmethod
    def parse(cls, parser: Parser) -> PlusCustomFormatOp:
        lhs = parser.parse_operand("Expected SSA Value name here!")
        parser.parse_characters("+", "Malformed operation format, expected `+`!")
        rhs = parser.parse_operand("Expected SSA Value name here!")
        parser.parse_punctuation(":")
        type = parser.parse_type()

        return PlusCustomFormatOp.create(operands=[lhs, rhs], result_types=[type])

    def print(self, printer: Printer):
        printer.print(" ", self.lhs, " + ", self.rhs, " : ", self.res.type)


def test_generic_format():
    """
    Test that we can use generic formats in operations.
    """
    prog = """
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "test.add"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
builtin.module {
  %0 = arith.constant 42 : i32
  %1 = test.add %0 + %0 : i32
}
"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None, False)


def test_custom_format():
    """
    Test that we can use custom formats in operations.
    """
    prog = """\
builtin.module {
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = test.add %0 + %0 : i32
}
"""

    expected = """\
builtin.module {
  %0 = arith.constant 42 : i32
  %1 = test.add %0 + %0 : i32
}
"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None, False)


def test_custom_format_II():
    """
    Test that we can print using generic formats.
    """
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = test.add %0 + %0 : i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "test.add"(%0, %0) : (i32, i32) -> i32
}) : () -> ()
"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None, print_generic_format=True)


@irdl_op_definition
class NoCustomFormatOp(IRDLOperation):
    name = "test.no_custom_format"

    ops: VarOperand = var_operand_def()
    res: VarOpResult = var_result_def()


def test_missing_custom_format():
    """
    Test that we can print using generic formats.
    """
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = test.no_custom_format(%0) : (i32) -> i32
}) : () -> ()
"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    with pytest.raises(ParseError):
        parser.parse_op()


@irdl_attr_definition
class CustomFormatAttr(ParametrizedAttribute):
    name = "custom"

    attr: ParameterDef[IntAttr]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        if parser.parse_optional_keyword("zero") is not None:
            parser.parse_characters(">")
            return [IntAttr(0)]
        if parser.parse_optional_keyword("one") is not None:
            parser.parse_characters(">")
            return [IntAttr(1)]
        assert False

    def print_parameters(self, printer: Printer) -> None:
        assert 0 <= self.attr.data <= 1
        printer.print("<", "zero" if self.attr.data == 0 else "one", ">")


class AnyOp(Operation):
    name = "any"


def test_custom_format_attr():
    """
    Test that we can parse and print attributes using custom formats.
    """
    prog = """\
"builtin.module"() ({
  "any"() {"attr" = #custom<zero>} : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  "any"() {"attr" = #custom<zero>} : () -> ()
}) : () -> ()"""

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_op(AnyOp)
    ctx.register_attr(CustomFormatAttr)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None)


def test_dictionary_attr():
    """Test that a DictionaryAttr can be parsed and then printed."""

    prog = """
"func.func"() {"sym_name" = "test", "function_type" = i64, "sym_visibility" = "private", "arg_attrs" = {"key_one"="value_one", "key_two"="value_two", "key_three"=72 : i64}} : () -> ()
    """

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)

    parser = Parser(ctx, prog)
    parsed = parser.parse_op()

    assert_print_op(parsed, prog, None)
