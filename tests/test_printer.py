from __future__ import annotations

from io import StringIO

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.arith import AddiOp, Arith, ConstantOp
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AnyFloat,
    Builtin,
    ComplexType,
    FloatAttr,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerType,
    ModuleOp,
    Signedness,
    SymbolRefAttr,
    UnitAttr,
    f32,
    i1,
    i32,
)
from xdsl.dialects.func import Func
from xdsl.ir import (
    Attribute,
    Block,
    Operation,
    ParametrizedAttribute,
    Region,
)
from xdsl.irdl import (
    IRDLOperation,
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
from xdsl.syntax_printer import SyntaxPrinter
from xdsl.utils.color_printer import ColorPrinter
from xdsl.utils.colors import Colors
from xdsl.utils.diagnostic import Diagnostic
from xdsl.utils.exceptions import ParseError
from xdsl.utils.test_value import create_ssa_value


def test_simple_forgotten_op():
    """Test that the parsing of an undefined operand gives it a name."""
    ctx = Context()
    ctx.load_dialect(Arith)

    lit = ConstantOp.from_int_and_width(42, 32)
    add = AddiOp(lit, lit)

    add.verify()

    expected = """%0 = "arith.addi"(%1, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32"""

    assert_print_op(add, expected)


def test_print_op_location():
    """Test that an op can be printed with its location."""
    ctx = Context()
    ctx.load_dialect(test.Test)

    add = test.TestOp(result_types=[i32])

    add.verify()

    expected = """%0 = "test.op"() : () -> i32 loc(unknown)"""

    assert_print_op(add, expected, print_debuginfo=True)


def test_print_location_types():
    """Test printing all location types."""
    from xdsl.dialects.builtin import (
        CallSiteLoc,
        FileLineColLoc,
        FileLineColRange,
        FusedLoc,
        NameLoc,
        StringAttr,
        UnknownLoc,
    )

    # FileLineColLoc
    loc = FileLineColLoc("test.cpp", 10, 8)
    stream = StringIO()
    Printer(stream).print_location(loc)
    assert stream.getvalue() == 'loc("test.cpp":10:8)'

    # FileLineColRange
    loc = FileLineColRange("test.cpp", 10, 8, 12, 18)
    stream = StringIO()
    Printer(stream).print_location(loc)
    assert stream.getvalue() == 'loc("test.cpp":10:8 to 12:18)'

    # NameLoc without child
    loc = NameLoc("CSE")
    stream = StringIO()
    Printer(stream).print_location(loc)
    assert stream.getvalue() == 'loc("CSE")'

    # NameLoc with child
    loc = NameLoc("CSE", FileLineColLoc("test.cpp", 10, 8))
    stream = StringIO()
    Printer(stream).print_location(loc)
    assert stream.getvalue() == 'loc("CSE"(loc("test.cpp":10:8)))'

    # CallSiteLoc
    loc = CallSiteLoc(UnknownLoc(), FileLineColLoc("main.cpp", 10, 8))
    stream = StringIO()
    Printer(stream).print_location(loc)
    assert stream.getvalue() == 'loc(callsite(loc(unknown) at loc("main.cpp":10:8)))'

    # FusedLoc without metadata
    loc = FusedLoc([FileLineColLoc("a.cpp", 1, 1), FileLineColLoc("b.cpp", 2, 2)])
    stream = StringIO()
    Printer(stream).print_location(loc)
    assert stream.getvalue() == 'loc(fused[loc("a.cpp":1:1), loc("b.cpp":2:2)])'

    # FusedLoc with metadata
    loc = FusedLoc([FileLineColLoc("a.cpp", 1, 1)], StringAttr("CSE"))
    stream = StringIO()
    Printer(stream).print_location(loc)
    assert stream.getvalue() == 'loc(fused<"CSE">[loc("a.cpp":1:1)])'


@irdl_op_definition
class UnitAttrOp(IRDLOperation):
    name = "unit_attr_op"

    parallelize = opt_attr_def(UnitAttr)


def test_unit_attr():
    """Test that a UnitAttr can be defined and printed"""

    expected = """
"unit_attr_op"() {parallelize} : () -> ()
"""

    unit_op = UnitAttrOp.build(attributes={"parallelize": UnitAttr()})

    assert_print_op(unit_op, expected)


def test_added_unit_attr():
    """Test that a UnitAttr can be added to an op, even if its not defined as a field."""

    expected = """
"unit_attr_op"() {parallelize, vectorize} : () -> ()
"""
    unitop = UnitAttrOp.build(
        attributes={"parallelize": UnitAttr(), "vectorize": UnitAttr()}
    )

    assert_print_op(unitop, expected)


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
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  ^^^^^^^^^^^^^^^^^^^^^
  | Test message
  ---------------------
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    first_op = module.ops.first
    assert first_op is not None
    diagnostic.add_message(first_op, "Test message")

    assert_print_op(module, expected, diagnostic=diagnostic)


def test_two_different_op_messages():
    """Test that an operation message can be printed."""
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  ^^^^^^^^^^^^^^^^^^^^^
  | Test message 1
  ---------------------
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  ^^^^^^^^^^^^^^^^^
  | Test message 2
  -----------------
}) : () -> ()"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    first_op, second_op = list(module.ops)
    diagnostic.add_message(first_op, "Test message 1")
    diagnostic.add_message(second_op, "Test message 2")

    assert_print_op(module, expected, diagnostic=diagnostic)


def test_two_same_op_messages():
    """Test that an operation message can be printed."""
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  ^^^^^^^^^^^^^^^^^^^^^
  | Test message 1
  ---------------------
  ^^^^^^^^^^^^^^^^^^^^^
  | Test message 2
  ---------------------
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    first_op, _second_op = list(module.ops)

    diagnostic.add_message(first_op, "Test message 1")
    diagnostic.add_message(first_op, "Test message 2")

    assert_print_op(module, expected, diagnostic=diagnostic)


def test_op_message_with_region():
    """Test that an operation message can be printed on an operation with a region."""
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
^^^^^^^^^^^^^^^^
| Test
----------------
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    diagnostic = Diagnostic()
    diagnostic.add_message(module, "Test")

    assert_print_op(module, expected, diagnostic=diagnostic)


def test_op_message_with_region_and_overflow():
    """
    Test that an operation message can be printed on an operation with a region,
    where the message is bigger than the operation.
    """
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
^^^^^^^^^^^^^^^^---
| Test long message
-------------------
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    diagnostic = Diagnostic()
    diagnostic.add_message(module, "Test long message")
    assert_print_op(module, expected, diagnostic=diagnostic)


def test_diagnostic():
    """
    Test that an operation message can be printed on an operation with a region,
    where the message is bigger than the operation.
    """
    prog = """\
"builtin.module"() ({
  %0 = arith.constant 42 : i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    class MyException(Exception): ...

    diag = Diagnostic()
    diag.add_message(module, "Test")
    with pytest.raises(MyException, match="Test"):
        diag.raise_exception(module, MyException("hello"))


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
  %213 = "arith.addi"(%i, %i) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %i = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %0 = "arith.addi"(%i, %i) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected)


def test_print_clashing_names():
    """
    Test the printer's value name printing logic's robustness against clashing names.

    This example now expects to print names i, i_1, i_2; it used to print i, i_1, i_1,
    printing a duplicate name for two values, meaning invalid IR as input for both MLIR
    and xDSL.
    """

    expected = """\
"builtin.module"() ({
  %i = "test.op"() : () -> i32
  %i_1 = "test.op"() : () -> i32
  %i_2 = "test.op"() : () -> i32
}) : () -> ()
"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)

    with ImplicitBuilder((module := ModuleOp([])).body):
        i = test.TestOp.create(result_types=[i32])
        i.results[0].name_hint = "i"
        j = test.TestOp.create(result_types=[i32])
        j.results[0].name_hint = "i"
        k = test.TestOp.create(result_types=[i32])
        k.results[0].name_hint = "i_1"

    assert_print_op(module, expected)


def test_print_custom_block_arg_name():
    block = Block(arg_types=[i32, i32])
    block.args[0].name_hint = "test"
    block.args[1].name_hint = "test"

    io = StringIO()
    p = Printer(stream=io)
    p.print_block(block)
    assert io.getvalue() == """\n^bb0(%test : i32, %test_1 : i32):"""


def test_print_block_argument():
    """Print a block argument."""
    block = Block(arg_types=[i32, i32])

    io = StringIO()
    p = Printer(stream=io)
    p.print_block_argument(block.args[0])
    p.print_string(", ")
    p.print_block_argument(block.args[1], print_type=False)
    assert io.getvalue() == """%0 : i32, %1"""


def test_print_block_argument_location():
    """Print a block argument with location."""
    block = Block(arg_types=[i32, i32])

    io = StringIO()
    p = Printer(stream=io, print_debuginfo=True)
    p.print_block_argument(block.args[0])
    p.print_string(", ")
    p.print_block_argument(block.args[1])
    # Block arguments default to UnknownLoc, which is printed with debuginfo
    assert io.getvalue() == """%0 : i32 loc(unknown), %1 : i32 loc(unknown)"""


def test_print_block():
    """Print a block."""
    block = Block(arg_types=[i32, i32])
    block.add_op(test.TestOp(operands=(block.args[1],)))

    # Print block arguments inside the block
    io = StringIO()
    p = Printer(stream=io)
    p.print_block(block)
    assert (
        io.getvalue()
        == """\n^bb0(%0 : i32, %1 : i32):\n  "test.op"(%1) : (i32) -> ()"""
    )


def test_print_block_without_arguments():
    """Print a block and its arguments separately."""
    block = Block(arg_types=[i32, i32])
    block.add_op(test.TestOp(operands=(block.args[1],)))

    # Print block arguments separately from the block
    io = StringIO()
    p = Printer(stream=io)
    p.print_block_argument(block.args[0])
    p.print_string(", ")
    p.print_block_argument(block.args[1])
    p.print_block(block, print_block_args=False)
    assert io.getvalue() == """%0 : i32, %1 : i32\n  "test.op"(%1) : (i32) -> ()"""


def test_print_block_with_terminator():
    """Print a block and with its terminator."""
    block = Block(ops=[test.TestOp.create(), test.TestTermOp.create()])

    # Print block ops including block terminator
    io = StringIO()
    p = Printer(stream=io)
    p.print_block(block, print_block_terminator=True)
    assert (
        io.getvalue()
        == """
^bb0:
  "test.op"() : () -> ()
  "test.termop"() : () -> ()"""
    )


def test_print_block_without_terminator():
    """Print a block and its terminator separately."""
    term_op = test.TestTermOp.create()
    block = Block(ops=[test.TestOp.create(), term_op])

    # Print block ops separately from the block terminator
    io = StringIO()
    p = Printer(stream=io)
    p.print_block(block, print_block_terminator=False)
    assert (
        io.getvalue()
        == """
^bb0:
  "test.op"() : () -> ()"""
    )


def test_print_region():
    """Print a region."""
    block = Block(arg_types=[i32, i32])
    block.add_op(test.TestOp(operands=(block.args[1],)))
    region = Region(block)

    io = StringIO()
    p = Printer(stream=io)
    p.print_region(region)
    assert (
        io.getvalue()
        == """{\n^bb0(%0 : i32, %1 : i32):\n  "test.op"(%1) : (i32) -> ()\n}"""
    )


def test_print_region_without_arguments():
    """Print a region and its arguments separately."""
    block = Block(arg_types=[i32, i32])
    block.add_op(test.TestOp(operands=(block.args[1],)))
    region = Region(block)

    io = StringIO()
    p = Printer(stream=io)
    p.print_block_argument(block.args[0])
    p.print_string(", ")
    p.print_block_argument(block.args[1])
    p.print_string(" ")
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
    assert io.getvalue() == """{\n^bb0(%0 : i32, %1 : i32):\n}"""


#   ____          _                  _____                          _
#  / ___|   _ ___| |_ ___  _ __ ___ |  ___|__  _ __ _ __ ___   __ _| |_
# | |  | | | / __| __/ _ \| '_ ` _ \| |_ / _ \| '__| '_ ` _ \ / _` | __|
# | |__| |_| \__ \ || (_) | | | | | |  _| (_) | |  | | | | | | (_| | |_
#  \____\__,_|___/\__\___/|_| |_| |_|_|  \___/|_|  |_| |_| |_|\__,_|\__|
#


@irdl_op_definition
class PlusCustomFormatOp(IRDLOperation):
    name = "test.add"
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType)

    @classmethod
    def parse(cls, parser: Parser) -> PlusCustomFormatOp:
        lhs = parser.parse_operand("Expected SSA Value name here!")
        parser.parse_characters("+", "Malformed operation format, expected `+`!")
        rhs = parser.parse_operand("Expected SSA Value name here!")
        parser.parse_punctuation(":")
        type = parser.parse_type()

        return PlusCustomFormatOp.create(operands=[lhs, rhs], result_types=[type])

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.lhs)
        printer.print_string(" + ")
        printer.print_ssa_value(self.rhs)
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


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

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, print_generic_format=False)


def test_custom_format():
    """
    Test that we can use custom formats in operations.
    """
    prog = """\
builtin.module {
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = test.add %0 + %0 : i32
}
"""

    expected = """\
builtin.module {
  %0 = arith.constant 42 : i32
  %1 = test.add %0 + %0 : i32
}
"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, print_generic_format=False)


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
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "test.add"(%0, %0) : (i32, i32) -> i32
}) : () -> ()
"""

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, print_generic_format=True)


@irdl_op_definition
class NoCustomFormatOp(IRDLOperation):
    name = "test.no_custom_format"

    ops = var_operand_def()
    res = var_result_def()


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

    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_op(NoCustomFormatOp)

    parser = Parser(ctx, prog)
    with pytest.raises(ParseError):
        parser.parse_op()


@irdl_attr_definition
class CustomFormatAttr(ParametrizedAttribute):
    name = "test.custom"

    attr: IntAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        if parser.parse_optional_keyword("zero") is not None:
            parser.parse_characters(">")
            return [IntAttr(0)]
        if parser.parse_optional_keyword("one") is not None:
            parser.parse_characters(">")
            return [IntAttr(1)]
        pytest.fail("zero or one expected")

    def print_parameters(self, printer: Printer) -> None:
        assert 0 <= self.attr.data <= 1
        with printer.in_angle_brackets():
            printer.print_string("zero" if self.attr.data == 0 else "one")


@irdl_op_definition
class AnyOp(IRDLOperation):
    name = "test.any"


def test_custom_format_attr():
    """
    Test that we can parse and print attributes using custom formats.
    """
    prog = """\
"builtin.module"() ({
  "test.any"() {attr = #test.custom<zero>} : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  "test.any"() {attr = #test.custom<zero>} : () -> ()
}) : () -> ()"""

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_op(AnyOp)
    ctx.load_attr_or_type(CustomFormatAttr)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected)


def test_dictionary_attr():
    """Test that a DictionaryAttr can be parsed and then printed."""

    prog = """
"func.func"() <{sym_name = "test", function_type = i64, sym_visibility = "private", unit_attr, arg_attrs = {key_one = "value_one", key_two = "value_two", key_three = 72 : i64, unit_attr}}> : () -> ()
    """

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)

    parser = Parser(ctx, prog)
    parsed = parser.parse_op()

    assert_print_op(parsed, prog)


def test_densearray_attr():
    """Test that a DenseArrayAttr can be parsed and then printed."""

    prog = """
"func.func"() <{sym_name = "test", function_type = i64, sym_visibility = "private", unit_attr}> {bool_attrs = array<i1: false, true>, int_attr = array<i32: 19, 23, 55>, float_attr = array<f32: 3.400000e-01>} : () -> ()
    """

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)

    parser = Parser(ctx, prog)
    parsed = parser.parse_op()

    assert_print_op(parsed, prog)


def test_float():
    printer = Printer()

    def _test_float_print(expected: str, value: float, type: AnyFloat):
        value = FloatAttr(value, type).value.data
        io = StringIO()
        printer.stream = io
        printer.print_float(value, type)
        assert io.getvalue() == expected

    _test_float_print("3.000000e+00", 3, f32)
    _test_float_print("-3.000000e+00", -3, f32)
    _test_float_print("3.140000e+00", 3.14, f32)
    _test_float_print("3.140000e+08", 3.14e8, f32)
    _test_float_print("3.14285707", 22 / 7, f32)
    _test_float_print("0x4D95DCF5", 22e8 / 7, f32)
    _test_float_print("3.14285714e+16", 22e16 / 7, f32)
    _test_float_print("-3.14285707", -22 / 7, f32)


@pytest.mark.parametrize(
    "expected, value, type",
    [
        ("true", -1, IntegerType(1)),
        ("false", 0, IntegerType(1)),
        ("true", True, IntegerType(1)),
        ("false", False, IntegerType(1)),
        ("-1", -1, IntegerType(1, signedness=Signedness.SIGNED)),
        ("0", 0, IntegerType(1, signedness=Signedness.SIGNED)),
        ("1", True, IntegerType(1, signedness=Signedness.SIGNED)),
        ("0", False, IntegerType(1, signedness=Signedness.SIGNED)),
        ("-1", -1, IntegerType(32)),
        ("0", 0, IntegerType(32)),
        ("1", True, IntegerType(32)),
        ("0", False, IntegerType(32)),
        ("-1", -1, IntegerType(32, signedness=Signedness.SIGNED)),
        ("0", 0, IntegerType(32, signedness=Signedness.SIGNED)),
        ("1", True, IntegerType(32, signedness=Signedness.SIGNED)),
        ("0", False, IntegerType(32, signedness=Signedness.SIGNED)),
        ("-1", -1, IndexType()),
        ("0", 0, IndexType()),
        ("1", True, IndexType()),
        ("0", False, IndexType()),
        ("-1", -1, None),
        ("0", 0, None),
        ("1", True, None),
        ("0", False, None),
    ],
)
def test_int(expected: str, value: int, type: IntegerType | IndexType | None):
    printer = Printer()
    printer.stream = StringIO()

    printer.print_int(value, type)

    assert printer.stream.getvalue() == expected


@pytest.mark.parametrize(
    "expected, value",
    [
        ("(-3.000000e+00,-3.000000e+00)", (-3.0, -3.0)),
        ("(3.000000e+00,3.000000e+00)", (3.0, 3.0)),
    ],
)
def test_complex_float(expected: str, value: tuple[float, float]):
    printer = Printer()
    io = StringIO()
    printer.stream = io
    type = ComplexType(f32)
    printer.print_complex_float(value, type)
    assert io.getvalue() == expected


@pytest.mark.parametrize(
    "expected, value",
    [
        ("(-3,-3)", (-3, -3)),
        ("(3,3)", (3, 3)),
    ],
)
def test_complex_int(expected: str, value: tuple[int, int]):
    printer = Printer()
    io = StringIO()
    printer.stream = io
    type = ComplexType(i32)
    printer.print_complex_int(value, type)
    assert io.getvalue() == expected


@pytest.mark.parametrize(
    "expected, value",
    [
        ("(true,true)", (1, 1)),
        ("(false,false)", (0, 0)),
    ],
)
def test_complex_bool(expected: str, value: tuple[int, int]):
    printer = Printer()
    io = StringIO()
    printer.stream = io
    type = ComplexType(i1)
    printer.print_complex_int(value, type)
    assert io.getvalue() == expected


@pytest.mark.parametrize(
    "expected, value, is_int",
    [
        ("(-3,-3)", (-3, -3), True),
        ("(3,3)", (3, 3), True),
        ("(-3.000000e+00,-3.000000e+00)", (-3.0, -3.0), False),
        ("(3.000000e+00,3.000000e+00)", (3.0, 3.0), False),
    ],
)
def test_complex(
    expected: str, value: tuple[int, int] | tuple[float, float], is_int: bool
):
    printer = Printer()
    io = StringIO()
    printer.stream = io
    if is_int:
        type = ComplexType(i32)
        printer.print_complex(value, type)
    else:
        type = ComplexType(f32)
        printer.print_complex(value, type)
    assert io.getvalue() == expected


def test_float_attr():
    printer = Printer()

    def _test_float_attr(value: float, type: AnyFloat):
        value = FloatAttr(value, type).value.data
        io_float = StringIO()
        printer.stream = io_float
        printer.print_float(value, type)

        io_attr = StringIO()
        printer.stream = io_attr
        FloatAttr(value, type).print_without_type(printer)

        assert io_float.getvalue() == io_attr.getvalue()

    for value in (
        3,
        3.14,
        22 / 7,
        float("nan"),
        float("inf"),
        float("-inf"),
    ):
        _test_float_attr(value, f32)


def test_float_attr_specials():
    printer = Printer()

    def _test_attr_print(expected: str, attr: FloatAttr):
        io = StringIO()
        printer.stream = io
        printer.print_attribute(attr)
        assert io.getvalue() == expected

    _test_attr_print("0x7e00 : f16", FloatAttr(float("nan"), 16))
    _test_attr_print("0x7c00 : f16", FloatAttr(float("inf"), 16))
    _test_attr_print("0xfc00 : f16", FloatAttr(float("-inf"), 16))

    _test_attr_print("0x7fc00000 : f32", FloatAttr(float("nan"), 32))
    _test_attr_print("0x7f800000 : f32", FloatAttr(float("inf"), 32))
    _test_attr_print("0xff800000 : f32", FloatAttr(float("-inf"), 32))

    _test_attr_print("0x7ff8000000000000 : f64", FloatAttr(float("nan"), 64))
    _test_attr_print("0x7ff0000000000000 : f64", FloatAttr(float("inf"), 64))
    _test_attr_print("0xfff0000000000000 : f64", FloatAttr(float("-inf"), 64))


@pytest.mark.parametrize(
    "dims, expected",
    [
        ([], ""),
        ([1, 2, 3], "1x2x3"),
        ([1, DYNAMIC_INDEX, 3, DYNAMIC_INDEX], "1x?x3x?"),
        ([5], "5"),
    ],
)
def test_print_dimension_list(dims: list[int], expected: str):
    io = StringIO()
    printer = Printer(stream=io)
    printer.print_dimension_list(dims)

    assert io.getvalue() == expected


def test_print_function_type():
    io = StringIO()
    printer = Printer(stream=io)
    printer.print_function_type((), ())

    assert io.getvalue() == "() -> ()"

    io = StringIO()
    printer.stream = io
    printer.print_function_type((i32,), ())

    assert io.getvalue() == "(i32) -> ()"

    io = StringIO()
    printer.stream = io
    printer.print_function_type((i32,), (i32,))

    assert io.getvalue() == "(i32) -> i32"

    io = StringIO()
    printer.stream = io
    printer.print_function_type((i32,), (i32, i32))

    assert io.getvalue() == "(i32) -> (i32, i32)"

    io = StringIO()
    printer.stream = io
    printer.print_function_type((i32,), (FunctionType.from_lists((i32,), (i32,)),))

    assert io.getvalue() == "(i32) -> ((i32) -> i32)"


def test_print_properties_as_attributes():
    """Test that properties can be printed as attributes."""

    prog = """
"func.func"() <{sym_name = "test", function_type = i64, sym_visibility = "private"}> {extra_attr} : () -> ()
    """

    retro_prog = """
"func.func"() {extra_attr, sym_name = "test", function_type = i64, sym_visibility = "private"} : () -> ()
    """

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)

    parser = Parser(ctx, prog)
    parsed = parser.parse_op()

    assert_print_op(parsed, retro_prog, print_properties_as_attributes=True)


def test_print_properties_as_attributes_safeguard():
    """Test that properties can be printed as attributes."""

    prog = """
"func.func"() <{sym_name = "test", function_type = i64, sym_visibility = "private"}> {extra_attr, sym_name = "this should be overriden by the property"} : () -> ()
    """

    retro_prog = """
"func.func"() {extra_attr, sym_name = "test", function_type = i64, sym_visibility = "private"} : () -> ()
    """

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)

    parser = Parser(ctx, prog)
    parsed = parser.parse_op()
    with pytest.raises(
        ValueError,
        match="Properties sym_name would overwrite the attributes of the same names.",
    ):
        assert_print_op(parsed, retro_prog, print_properties_as_attributes=True)


@pytest.mark.parametrize(
    "attr,expected",
    [
        (SymbolRefAttr("foo"), "@foo"),
        (SymbolRefAttr("weird name!!"), '@"weird name!!"'),
        (
            SymbolRefAttr("weird nested", ["yes", "very nested"]),
            '@"weird nested"::@yes::@"very nested"',
        ),
    ],
)
def test_symbol_ref(attr: SymbolRefAttr, expected: str):
    ctx = Context()
    ctx.load_dialect(Builtin)

    printed = StringIO()
    Printer(printed).print_attribute(attr)
    assert printed.getvalue() == expected


def test_get_printed_name():
    ctx = Context()
    ctx.load_dialect(Builtin)

    printer = Printer()
    val = create_ssa_value(i32)

    # Test printing without constraints
    stream = StringIO()
    printer.stream = stream
    picked_name = printer.print_ssa_value(val)
    assert f"%{picked_name}" == printer.stream.getvalue()

    # Test printing when name has already been picked
    stream = StringIO()
    printer.stream = stream
    picked_name = printer.print_ssa_value(val)
    assert f"%{picked_name}" == printer.stream.getvalue()

    # Test printing with name hint
    val = create_ssa_value(i32)
    val.name_hint = "foo"
    printed = StringIO()
    picked_name = Printer(printed).print_ssa_value(val)
    assert f"%{picked_name}" == printed.getvalue()


def test_delimiters():
    printer = Printer()

    printer.stream = StringIO()
    with printer.in_angle_brackets():
        printer.print_string("testing")
    assert "<testing>" == printer.stream.getvalue()

    printer.stream = StringIO()
    with printer.in_square_brackets():
        printer.print_string("testing")
    assert "[testing]" == printer.stream.getvalue()

    printer.stream = StringIO()
    with printer.in_braces():
        printer.print_string("testing")
    assert "{testing}" == printer.stream.getvalue()

    printer.stream = StringIO()
    with printer.in_parens():
        printer.print_string("testing")
    assert "(testing)" == printer.stream.getvalue()

    printer.stream = StringIO()
    with printer.delimited("test<", ">"):
        printer.print_string("testing")
    assert "test<testing>" == printer.stream.getvalue()


def test_symbol_printing():
    printer = Printer()

    printer.stream = StringIO()
    printer.print_symbol_name("symbol")
    assert "@symbol" == printer.stream.getvalue()

    printer.stream = StringIO()
    printer.print_symbol_name("@symbol")
    assert '@"@symbol"' == printer.stream.getvalue()


def test_color_printing():
    printer = ColorPrinter()

    printer.stream = StringIO()
    with printer.colored(None):
        printer.print_string("test")
    assert "test" == printer.stream.getvalue()

    printer.stream = StringIO()
    with printer.colored(Colors.BLUE):
        printer.print_string("test")
    assert "\x1b[34mtest\x1b[0m" == printer.stream.getvalue()


def test_nested_color_printing():
    printer = ColorPrinter()

    printer.stream = StringIO()
    with printer.colored(Colors.BLUE):
        with printer.colored(Colors.RED):
            printer.print_string("test")

    # Should still be blue
    assert "\x1b[34mtest\x1b[0m" == printer.stream.getvalue()

    printer.stream = StringIO()
    with printer.colored(None):
        with printer.colored(Colors.RED):
            printer.print_string("test")

    # Should be red
    assert "\x1b[31mtest\x1b[0m" == printer.stream.getvalue()


def test_syntax_printer():
    printer = SyntaxPrinter()

    op = test.TestOp(result_types=(IntegerType(32),))

    printer.stream = StringIO()
    printer.print_ssa_value(op.results[0])

    assert "\x1b[95m%0\x1b[0m" == printer.stream.getvalue()


def assert_print_op(
    operation: Operation,
    expected: str,
    *,
    diagnostic: Diagnostic | None = None,
    print_generic_format: bool = True,
    print_debuginfo: bool = False,
    print_properties_as_attributes: bool = False,
    indent_num_spaces: int = 2,
):
    """
    Utility function that helps to check the printing of an operation compared to
    some string.

    ### Example:

    To check that an operation, e.g. `arith.addi` prints as expected:

    .. code-block:: py
        expected = \"\"\"

        builtin.module() {
        %0 : !i32 = arith.addi(%<UNKNOWN> : !i32, %<UNKNOWN> : !i32)
        -----------------------^^^^^^^^^^----------------------------------------------------------------
        | ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
        -------------------------------------------------------------------------------------------------
        ------------------------------------------^^^^^^^^^^---------------------------------------------
        | ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
        -------------------------------------------------------------------------------------------------
        %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
        }\"\"\"


    we call:

    .. code-block:: python

        assert_print_op(add, expected)

    Additional options can be passed to the printer using keyword arguments:

    .. code-block:: python

        assert_print_op(add, expected, indent_num_spaces=4)

    """

    file = StringIO("")
    if diagnostic is None:
        diagnostic = Diagnostic()
    printer = Printer(
        stream=file,
        print_generic_format=print_generic_format,
        print_properties_as_attributes=print_properties_as_attributes,
        print_debuginfo=print_debuginfo,
        diagnostic=diagnostic,
        indent_num_spaces=indent_num_spaces,
    )

    printer.print_op(operation)
    assert file.getvalue().strip() == expected.strip()
