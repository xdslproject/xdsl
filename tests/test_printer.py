from __future__ import annotations

import re
import pytest
from io import StringIO
from typing import Annotated

from xdsl.dialects.arith import Arith, Addi, Constant
from xdsl.dialects.builtin import Builtin, IntAttr, IntegerType, ModuleOp, UnitAttr, i32
from xdsl.dialects.func import Func
from xdsl.ir import (
    Attribute,
    MLContext,
    OpResult,
    Operation,
    ParametrizedAttribute,
    Block,
)
from xdsl.irdl import (
    Operand,
    OptOpAttr,
    ParameterDef,
    VarOpResult,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    IRDLOperation,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.diagnostic import Diagnostic

from conftest import assert_print_op
from xdsl.utils.exceptions import ParseError


def test_simple_forgotten_op():
    """Test that the parsing of an undefined operand raises an exception."""
    ctx = MLContext()
    ctx.register_dialect(Arith)

    lit = Constant.from_int_and_width(42, 32)
    add = Addi(lit, lit)

    add.verify()

    expected = """
%0 = "arith.addi"(%<UNKNOWN>, %<UNKNOWN>) : (i32, i32) -> i32
------------------^^^^^^^^^^---------------------------------------------------------------------
| ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
-------------------------------------------------------------------------------------------------
------------------------------^^^^^^^^^^---------------------------------------------------------
| ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
-------------------------------------------------------------------------------------------------
"""

    assert_print_op(add, expected, None)


def test_forgotten_op_non_fail():
    """Test that the parsing of an undefined operand raises an exception."""
    ctx = MLContext()
    ctx.register_dialect(Arith)

    lit = Constant.from_int_and_width(42, 32)
    add = Addi(lit, lit)
    add2 = Addi(add, add)
    mod = ModuleOp([add, add2])
    mod.verify()

    expected = """
"builtin.module"() ({
  %0 = "arith.addi"(%<UNKNOWN>, %<UNKNOWN>) : (i32, i32) -> i32
  ------------------^^^^^^^^^^---------------------------------------------------------------------
  | ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
  -------------------------------------------------------------------------------------------------
  ------------------------------^^^^^^^^^^---------------------------------------------------------
  | ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
  -------------------------------------------------------------------------------------------------
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    assert_print_op(mod, expected, None)


@irdl_op_definition
class UnitAttrOp(IRDLOperation):
    name = "unit_attr_op"

    parallelize: OptOpAttr[UnitAttr]


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
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
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
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
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
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
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
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
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
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
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
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
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
  %i = "arith.constant"() {"value" = 42 : i32} : () -> i32
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
    assert io.getvalue() == """\n^0(%test : i32, %0 : i32):"""


#   ____          _                  _____                          _
#  / ___|   _ ___| |_ ___  _ __ ___ |  ___|__  _ __ _ __ ___   __ _| |_
# | |  | | | / __| __/ _ \| '_ ` _ \| |_ / _ \| '__| '_ ` _ \ / _` | __|
# | |__| |_| \__ \ || (_) | | | | | |  _| (_) | |  | | | | | | (_| | |_
#  \____\__,_|___/\__\___/|_| |_| |_|_|  \___/|_|  |_| |_| |_|\__,_|\__|
#


@irdl_op_definition
class PlusCustomFormatOp(IRDLOperation):
    name = "test.add"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    res: Annotated[OpResult, IntegerType]

    @classmethod
    def parse(cls, parser: Parser) -> PlusCustomFormatOp:
        lhs = parser.parse_operand("Expected SSA Value name here!")
        parser.parse_characters("+", "Malformed operation format, expected `+`!")
        rhs = parser.parse_operand("Expected SSA Value name here!")
        parser.parse_punctuation(":")
        type = parser.expect(parser.try_parse_type, "Expect type here!")

        return PlusCustomFormatOp.create(operands=[lhs, rhs], result_types=[type])

    def print(self, printer: Printer):
        printer.print(" ", self.lhs, " + ", self.rhs, " : ", self.res.typ)


def test_generic_format():
    """
    Test that we can use generic formats in operations.
    """
    prog = """
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "test.add"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = test.add %0 + %0 : i32
}) : () -> ()
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
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = test.add %0 + %0 : i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = test.add %0 + %0 : i32
}) : () -> ()
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
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
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

    ops: VarOperand
    res: VarOpResult


def test_missing_custom_format():
    """
    Test that we can print using generic formats.
    """
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
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

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_char("<")
        value = parser.tokenizer.next_token_of_pattern(re.compile("(zero|one)"))
        if value and value.text == "zero":
            parser.parse_char(">")
            return [IntAttr(0)]
        if value and value.text == "one":
            parser.parse_char(">")
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
