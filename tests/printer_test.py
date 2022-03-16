from io import StringIO

from xdsl.printer import Printer
from xdsl.parser import Parser
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.arith import *
from xdsl.diagnostic import Diagnostic


def test_forgotten_op():
    """Test that the parsing of an undefined operand raises an exception."""
    ctx = MLContext()
    arith = Arith(ctx)

    lit = Constant.from_int_constant(42, 32)
    add = Addi.get(lit, lit)

    add.verify()
    try:
        printer = Printer()
        printer.print_op(add)
    except KeyError:
        return

    assert False, "Exception expected"


def test_op_message():
    """Test that an operation message can be printed."""
    prog = \
        """module() {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
    }"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message
  --------------------------------------------------
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

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
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message 1
  --------------------------------------------------
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message 2
  --------------------------------------------
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
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message 1
  --------------------------------------------------
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  | Test message 2
  --------------------------------------------------
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
^^^^^^^^
| Test
--------
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
