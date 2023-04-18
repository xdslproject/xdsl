from __future__ import annotations

import re
from io import StringIO
from typing import List, Annotated

from xdsl.dialects.arith import Arith, Addi, Constant
from xdsl.dialects.builtin import Builtin, IntAttr, ModuleOp, IntegerType, UnitAttr, i32
from xdsl.dialects.func import Func
from xdsl.ir import Attribute, MLContext, OpResult, Operation, ParametrizedAttribute, Block
from xdsl.irdl import (OptOpAttr, ParameterDef, irdl_attr_definition,
                       irdl_op_definition, IRDLOperation, Operand)
from xdsl.parser import Parser, BaseParser, XDSLParser
from xdsl.printer import Printer
from xdsl.utils.diagnostic import Diagnostic

from conftest import assert_print_op


def test_simple_forgotten_op():
    """Test that the parsing of an undefined operand raises an exception."""
    ctx = MLContext()
    ctx.register_dialect(Arith)

    lit = Constant.from_int_and_width(42, 32)
    add = Addi.get(lit, lit)

    add.verify()

    expected = \
"""
%0 : !i32 = arith.addi(%<UNKNOWN> : !i32, %<UNKNOWN> : !i32)
-----------------------^^^^^^^^^^----------------------------------------------------------------
| ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
-------------------------------------------------------------------------------------------------
------------------------------------------^^^^^^^^^^---------------------------------------------
| ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
-------------------------------------------------------------------------------------------------
"""

    assert_print_op(add, expected, None)


def test_forgotten_op_non_fail():
    """Test that the parsing of an undefined operand raises an exception."""
    ctx = MLContext()
    ctx.register_dialect(Arith)

    lit = Constant.from_int_and_width(42, 32)
    add = Addi.get(lit, lit)
    add2 = Addi.get(add, add)
    mod = ModuleOp.from_region_or_ops([add, add2])
    mod.verify()

    expected = \
"""
builtin.module() {
  %0 : !i32 = arith.addi(%<UNKNOWN> : !i32, %<UNKNOWN> : !i32)
  -----------------------^^^^^^^^^^----------------------------------------------------------------
  | ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
  -------------------------------------------------------------------------------------------------
  ------------------------------------------^^^^^^^^^^---------------------------------------------
  | ERROR: SSAValue is not part of the IR, are you sure all operations are added before their uses?
  -------------------------------------------------------------------------------------------------
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}
"""

    assert_print_op(mod, expected, None)


@irdl_op_definition
class UnitAttrOp(IRDLOperation):
    name = "unit_attr_op"

    parallelize: OptOpAttr[UnitAttr]


def test_unit_attr():
    """Test that a UnitAttr can be defined and printed"""

    expected = \
"""
unit_attr_op() ["parallelize"]
"""

    unit_op = UnitAttrOp.build(attributes={"parallelize": UnitAttr([])})

    assert_print_op(unit_op, expected, None)


def test_added_unit_attr():
    """Test that a UnitAttr can be added to an op, even if its not defined as a field."""

    expected = \
"""
unit_attr_op() ["parallelize", "vectorize"]
"""
    unitop = UnitAttrOp.build(attributes={
        "parallelize": UnitAttr([]),
        "vectorize": UnitAttr([])
    })

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
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""
builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message
  --------------------------
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}
"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    diagnostic.add_message(module.ops[0], "Test message")

    assert_print_op(module, expected, diagnostic)


def test_two_different_op_messages():
    """Test that an operation message can be printed."""
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message 1
  --------------------------
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
  ^^^^^^^^^^^^^^^^^^^^^^
  | Test message 2
  ----------------------
}"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    diagnostic.add_message(module.ops[0], "Test message 1")
    diagnostic.add_message(module.ops[1], "Test message 2")

    assert_print_op(module, expected, diagnostic)


def test_two_same_op_messages():
    """Test that an operation message can be printed."""
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message 1
  --------------------------
  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message 2
  --------------------------
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_module()

    diagnostic = Diagnostic()
    diagnostic.add_message(module.ops[0], "Test message 1")
    diagnostic.add_message(module.ops[0], "Test message 2")

    assert_print_op(module, expected, diagnostic)


def test_op_message_with_region():
    """Test that an operation message can be printed on an operation with a region."""
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""\
builtin.module() {
^^^^^^^^^^^^^^
| Test
--------------
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_op()

    diagnostic = Diagnostic()
    diagnostic.add_message(module, "Test")

    assert_print_op(module, expected, diagnostic)


def test_op_message_with_region_and_overflow():
    """
    Test that an operation message can be printed on an operation with a region,
    where the message is bigger than the operation.
    """
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""\
builtin.module() {
^^^^^^^^^^^^^^-----
| Test long message
-------------------
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_op()

    diagnostic = Diagnostic()
    diagnostic.add_message(module, "Test long message")
    assert_print_op(module, expected, diagnostic)


def test_diagnostic():
    """
    Test that an operation message can be printed on an operation with a region,
    where the message is bigger than the operation.
    """
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = XDSLParser(ctx, prog)
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
    prog = \
        """builtin.module() {
    %i : !i32 = arith.constant() ["value" = 42 : !i32]
    %213 : !i32 = arith.addi(%i : !i32, %i : !i32)
    }"""

    expected = \
"""\
builtin.module() {
  %i : !i32 = arith.constant() ["value" = 42 : !i32]
  %0 : !i32 = arith.addi(%i : !i32, %i : !i32)
}"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None)


def test_print_custom_block_arg_name():
    block = Block(arg_types=[i32, i32])
    block.args[0].name = "test"
    block.args[1].name = "test"

    io = StringIO()
    p = Printer(target=Printer.Target.MLIR, stream=io)
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
    def parse(cls, result_types: List[Attribute],
              parser: BaseParser) -> PlusCustomFormatOp:
        lhs = parser.parse_operand('Expected SSA Value name here!')
        parser.parse_characters("+",
                                "Malformed operation format, expected `+`!")
        rhs = parser.parse_operand('Expected SSA Value name here!')

        return PlusCustomFormatOp.create(operands=[lhs, rhs],
                                         result_types=result_types)

    def print(self, printer: Printer):
        printer.print(" ", self.lhs, " + ", self.rhs)


def test_generic_format():
    """
    Test that we can use generic formats in operations.
    """
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = "test.add"(%0: !i32, %0: !i32)
    }"""

    expected = \
"""\
builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = test.add %0 + %0
}"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_op(PlusCustomFormatOp)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None)


def test_custom_format():
    """
    Test that we can use custom formats in operations.
    """
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = test.add %0 + %0
    }"""

    expected = \
"""\
builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = test.add %0 + %0
}"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_op(PlusCustomFormatOp)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None)


def test_custom_format_II():
    """
    Test that we can print using generic formats.
    """
    prog = \
        """builtin.module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = test.add %0 + %0
    }"""

    expected = \
"""\
"builtin.module"() {
  %0 : !i32 = "arith.constant"() ["value" = 42 : !i32]
  %1 : !i32 = "test.add"(%0 : !i32, %0 : !i32)
}"""

    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_op(PlusCustomFormatOp)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None, print_generic_format=True)


@irdl_attr_definition
class CustomFormatAttr(ParametrizedAttribute):
    name = "custom"

    attr: ParameterDef[IntAttr]

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        parser.parse_char("<")
        value = parser.tokenizer.next_token_of_pattern(
            re.compile('(zero|one)'))
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
    prog = \
        """builtin.module() {
      any() ["attr" = !custom<zero>]
    }"""

    expected = \
"""\
builtin.module() {
  any() ["attr" = !custom<zero>]
}"""

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_op(AnyOp)
    ctx.register_attr(CustomFormatAttr)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None)


def test_parse_generic_format_attr_II():
    """
    Test that we can parse attributes using generic formats.
    """
    prog = \
        """builtin.module() {
      any() ["attr" = !custom<zero>]
    }"""

    expected = \
"""\
"builtin.module"() {
  "any"() ["attr" = !"custom"<!int<0>>]
}"""

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_op(AnyOp)
    ctx.register_attr(CustomFormatAttr)

    parser = XDSLParser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None, print_generic_format=True)


def test_parse_generic_format_attr_III():
    """
    Test that we can parse attributes using generic formats.
    """
    prog = \
        """builtin.module() {
      any() ["attr" = !custom<one>]
    }"""

    expected = \
"""\
"builtin.module"() {
  "any"() ["attr" = !"custom"<!int<1>>]
}"""

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_op(AnyOp)
    ctx.register_attr(CustomFormatAttr)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    assert_print_op(module, expected, None, print_generic_format=True)


def test_foo_string():
    """
    Fail attribute in purpose.
    """
    prog = \
        """builtin.module() {
      any() ["attr" = !"string"<"foo">]
    }"""

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_op(AnyOp)
    ctx.register_attr(CustomFormatAttr)

    parser = XDSLParser(ctx, prog)
    try:
        parser.parse_op()
        assert False
    except:
        pass


def test_dictionary_attr():
    """Test that a DictionaryAttr can be parsed and then printed."""

    prog = """
    func.func() ["sym_name" = "test", "function_type" = !i64, "sym_visibility" = "private", "arg_attrs" = {"key_one"="value_one", "key_two"="value_two", "key_three"=72 : !i64}]
    """

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)

    parser = XDSLParser(ctx, prog)
    parsed = parser.parse_op()

    assert_print_op(parsed, prog, None)
