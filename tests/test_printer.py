from __future__ import annotations

from io import StringIO
from typing import List, Annotated

from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.builtin import Builtin, IntAttr, ModuleOp, IntegerType, UnitAttr
from xdsl.dialects.arith import Arith, Addi, Constant

from xdsl.ir import Attribute, MLContext, OpResult, ParametrizedAttribute, SSAValue
from xdsl.irdl import (ParameterDef, irdl_attr_definition, irdl_op_definition,
                       Operation, OperandDef, ResultDef, OptAttributeDef)
from xdsl.printer import Printer
from xdsl.parser import Parser
from xdsl.utils.diagnostic import Diagnostic


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

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(add)

    assert file.getvalue().strip() == expected.strip()


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

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(mod)

    assert file.getvalue().strip() == expected.strip()


@irdl_op_definition
class UnitAttrOp(Operation):
    name = "unit_attr_op"

    parallelize = OptAttributeDef(UnitAttr)


def test_unit_attr():
    """Test that a UnitAttr can be defined and printed"""

    expected = \
"""
unit_attr_op() ["parallelize"]
"""

    file = StringIO("")
    printer = Printer(stream=file)

    unit_op = UnitAttrOp.build(attributes={"parallelize": UnitAttr([])})

    printer.print_op(unit_op)
    assert file.getvalue().strip() == expected.strip()


def test_added_unit_attr():
    """Test that a UnitAttr can be added to an op, even if its not defined as a field."""

    expected = \
"""
unit_attr_op() ["parallelize", "vectorize"]
"""
    file = StringIO("")
    printer = Printer(stream=file)
    unitop = UnitAttrOp.build(attributes={
        "parallelize": UnitAttr([]),
        "vectorize": UnitAttr([])
    })

    printer.print_op(unitop)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    diagnostic = Diagnostic()
    diagnostic.add_message(module.ops[0], "Test message")
    printer = Printer(stream=file, diagnostic=diagnostic)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    diagnostic = Diagnostic()
    diagnostic.add_message(module.ops[0], "Test message 1")
    diagnostic.add_message(module.ops[1], "Test message 2")
    printer = Printer(stream=file, diagnostic=diagnostic)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    diagnostic = Diagnostic()
    printer = Printer(stream=file, diagnostic=diagnostic)
    diagnostic.add_message(module.ops[0], "Test message 1")
    diagnostic.add_message(module.ops[0], "Test message 2")
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    diagnostic = Diagnostic()
    printer = Printer(stream=file, diagnostic=diagnostic)
    diagnostic.add_message(module, "Test")
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    diagnostic = Diagnostic()
    printer = Printer(stream=file, diagnostic=diagnostic)
    diagnostic.add_message(module, "Test long message")
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


#   ____          _                  _____                          _
#  / ___|   _ ___| |_ ___  _ __ ___ |  ___|__  _ __ _ __ ___   __ _| |_
# | |  | | | / __| __/ _ \| '_ ` _ \| |_ / _ \| '__| '_ ` _ \ / _` | __|
# | |__| |_| \__ \ || (_) | | | | | |  _| (_) | |  | | | | | | (_| | |_
#  \____\__,_|___/\__\___/|_| |_| |_|_|  \___/|_|  |_| |_| |_|\__,_|\__|
#


@irdl_op_definition
class PlusCustomFormatOp(Operation):
    name = "test.add"
    lhs: Annotated[SSAValue, OperandDef(IntegerType)]
    rhs: Annotated[SSAValue, OperandDef(IntegerType)]
    res: Annotated[OpResult, ResultDef(IntegerType)]

    @classmethod
    def parse(cls, result_types: List[Attribute],
              parser: Parser) -> PlusCustomFormatOp:
        lhs = parser.parse_ssa_value()
        parser.skip_white_space()
        parser.parse_char("+")
        rhs = parser.parse_ssa_value()
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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file, print_generic_format=True)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


@irdl_attr_definition
class CustomFormatAttr(ParametrizedAttribute):
    name = "custom"

    attr: ParameterDef[IntAttr]

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_char("<")
        value = parser.parse_alpha_num(skip_white_space=False)
        if value == "zero":
            parser.parse_char(">")
            return [IntAttr.from_int(0)]
        if value == "one":
            parser.parse_char(">")
            return [IntAttr.from_int(1)]
        assert False

    def print_parameters(self, printer: Printer) -> None:
        assert 0 <= self.attr.data <= 1
        printer.print("<", "zero" if self.attr.data == 0 else "one", ">")


@irdl_op_definition
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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


def test_parse_generic_format_attr():
    """
    Test that we can parse attributes using generic formats.
    """
    prog = \
        """builtin.module() {
      any() ["attr" = !"custom"<!int<0>>]
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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file, print_generic_format=True)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


def test_parse_dense_xdsl():
    '''
    Test that parsing of shaped dense tensors works.
    '''
    # TODO: handle nested array syntax
    prog = """
    %0 : !tensor<[2 : !index, 3 : !index], !f64> = arith.constant() ["value" = !dense<!tensor<[2 : !index, 3 : !index], !f64>, [1.0 : !f64, 2.0 : !f64, 3.0 : !f64, 4.0 : !f64, 5.0 : !f64, 6.0 : !f64]>]
    """

    expected = """
    %0 : !tensor<[2 : !index, 3 : !index], !f64> = arith.constant() ["value" = !dense<!tensor<[2 : !index, 3 : !index], !f64>, [1.0 : !f64, 2.0 : !f64, 3.0 : !f64, 4.0 : !f64, 5.0 : !f64, 6.0 : !f64]>]
    """

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Arith)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


def test_parse_dense_mlir():
    """
    Test that we can parse attributes using generic formats.
    """
    prog = """
    %0 = "arith.constant"() {"value" = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    """

    expected = """
    %0 = "arith.constant"() {"value" = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    """

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Arith)

    parser = Parser(ctx, prog, source=Parser.Source.MLIR)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file, target=Printer.Target.MLIR)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


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

    parser = Parser(ctx, prog)
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

    expected = """
    func.func() ["sym_name" = "test", "function_type" = !i64, "sym_visibility" = "private", "arg_attrs" = {"key_one"="value_one", "key_two"="value_two", "key_three"=72 : !i64}]
    """

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)

    parser = Parser(ctx, prog)
    parsed = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(parsed)
    assert file.getvalue().strip() == expected.strip()
