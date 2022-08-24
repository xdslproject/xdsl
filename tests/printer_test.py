from __future__ import annotations

from io import StringIO

from xdsl.dialects.builtin import Builtin, ModuleOp, IntegerType
from xdsl.dialects.arith import Arith, Addi, Constant
from xdsl.diagnostic import Diagnostic
from xdsl.ir import MLContext
from xdsl.irdl import irdl_op_definition, Operation, OperandDef, ResultDef
from xdsl.printer import Printer
from xdsl.parser import Parser


def test_simple_forgotten_op():
    """Test that the parsing of an undefined operand raises an exception."""
    ctx = MLContext()
    arith = Arith(ctx)

    lit = Constant.from_int_constant(42, 32)
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
    arith = Arith(ctx)

    lit = Constant.from_int_constant(42, 32)
    add = Addi.get(lit, lit)
    add2 = Addi.get(add, add)
    mod = ModuleOp.from_region_or_ops([add, add2])
    mod.verify()

    expected = \
"""
module() {
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
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""
module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message
  --------------------------
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}
"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)

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
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""module() {
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
    arith = Arith(ctx)
    builtin = Builtin(ctx)

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
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""module() {
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
    arith = Arith(ctx)
    builtin = Builtin(ctx)

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
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""\
module() {
^^^^^^
| Test
------
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)

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
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""\
module() {
^^^^^^--------
| Test message
--------------
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    diagnostic = Diagnostic()
    printer = Printer(stream=file, diagnostic=diagnostic)
    diagnostic.add_message(module, "Test message")
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


def test_diagnostic():
    """
    Test that an operation message can be printed on an operation with a region,
    where the message is bigger than the operation.
    """
    prog = \
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""\
Exception: test message

module() {
^^^^^^^^-------
| Test message
---------------
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)

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


def test_print_costum_name():
    """
    Test that an SSAValue, that is a name and not a number, reserves that name
    """
    prog = \
        """module() {
    %i : !i32 = arith.constant() ["value" = 42 : !i32]
    %213 : !i32 = arith.addi(%i : !i32, %i : !i32)
    }"""

    expected = \
"""\
module() {
  %i : !i32 = arith.constant() ["value" = 42 : !i32]
  %0 : !i32 = arith.addi(%i : !i32, %i : !i32)
}"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)

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
    lhs = OperandDef(IntegerType)
    rhs = OperandDef(IntegerType)
    res = ResultDef(IntegerType)

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
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = "test.add"(%0: !i32, %0: !i32)
    }"""

    expected = \
"""\
module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = test.add %0 + %0
}"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)
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
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = test.add %0 + %0
    }"""

    expected = \
"""\
module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = test.add %0 + %0
}"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)
    ctx.register_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()


def test_custom_format():
    """
    Test that we can print using generic formats.
    """
    prog = \
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = test.add %0 + %0
    }"""

    expected = \
"""\
"module"() {
  %0 : !i32 = "arith.constant"() ["value" = 42 : !i32]
  %1 : !i32 = "test.add"(%0 : !i32, %0 : !i32)
}"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)
    ctx.register_op(PlusCustomFormatOp)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file, print_generic_format=True)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()
