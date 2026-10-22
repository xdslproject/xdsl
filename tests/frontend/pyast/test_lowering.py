from collections.abc import Sequence
from textwrap import dedent

import pytest

from xdsl.context import Context
from xdsl.dialects import arith, test
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, FunctionType, IntegerAttr, i32
from xdsl.dialects.func import Func
from xdsl.dialects.py import Py
from xdsl.dialects.scf import Scf
from xdsl.dialects.symref import Symref
from xdsl.dialects.test import Test
from xdsl.frontend.pyast.lowering import (
    FunctionDefinitionConstructor,
    PyLowering,
    ReturnConstructor,
    build_func,
    build_return,
)
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.ir import Region, SSAValue
from xdsl.parser import Parser
from xdsl.utils.lexer import Location


def effect(value: int) -> int: ...
def consume(value: int) -> None: ...
def pair(lhs: int, rhs: int) -> tuple[int, int]: ...
def unregistered() -> None: ...


class Counter:
    @classmethod
    def start(cls) -> int: ...

    @classmethod
    def stop(cls) -> None: ...


def build_constant(value: int) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(value, i32))


def build_less_than(lhs: SSAValue, rhs: SSAValue) -> arith.CmpiOp:
    return arith.CmpiOp(lhs, rhs, "slt")


def build_effect(value: SSAValue) -> test.TestOp:
    return test.TestOp([value], [i32])


def build_consume(value: SSAValue) -> test.TestOp:
    return test.TestOp([value])


def build_pair(lhs: SSAValue, rhs: SSAValue) -> test.TestOp:
    return test.TestOp([lhs, rhs], [i32, i32])


def build_start() -> test.TestOp:
    return test.TestOp(result_types=[i32])


def lower(
    program: str,
    function_definition_constructor: FunctionDefinitionConstructor = build_func,
    return_constructor: ReturnConstructor = build_return,
) -> str:
    ctx = Context()
    for dialect in (Arith, Builtin, Func, Py, Scf, Symref, Test):
        ctx.load_dialect(dialect)
    module = Parser(ctx, program).parse_module()

    type_converter = TypeConverter(globals())
    type_converter.type_registry.insert(int, i32)
    type_converter.literal_registry.insert(int, build_constant)
    type_converter.function_registry.insert(int.__add__, arith.AddiOp)
    type_converter.function_registry.insert(int.__lt__, build_less_than)
    type_converter.function_registry.insert(effect, build_effect)
    type_converter.function_registry.insert(consume, build_consume)
    type_converter.function_registry.insert(pair, build_pair)
    type_converter.function_registry.insert(Counter.start, build_start)

    PyLowering(
        type_converter, "test.py", function_definition_constructor, return_constructor
    ).lower(module)
    return str(module)


def test_lower_function():
    program = """
    py.func @f ["int", "int"] -> "int" {
    ^bb0(%a: !py.object, %b: !py.object):
      symref.declare "a"
      symref.update @a = %a : !py.object
      %0 = symref.fetch @a : !py.object
      %1 = py.compare "lt" %0 %b
      %2 = py.bool %1
      %3 = scf.if %2 -> (!py.object) {
        %4 = py.const 1
        scf.yield %4 : !py.object
      } else {
        %5 = py.binop "add" %0 %b
        scf.yield %5 : !py.object
      }
      %6 = py.call "effect"(%3) keywords ["value"]
      symref.update @a = %6 : !py.object
      scf.if %2 {
        %7 = py.call "consume"(%6)
        scf.yield
      } else {
        scf.yield
      }
      py.return %6
    }
    """
    assert lower(program) == dedent("""\
        builtin.module {
          func.func @f(%a: i32, %b: i32) -> i32 {
            symref.declare "a"
            symref.update @a = %a : i32
            %0 = symref.fetch @a : i32
            %1 = arith.cmpi slt, %0, %b : i32
            %2 = scf.if %1 -> (i32) {
              %3 = arith.constant 1 : i32
              scf.yield %3 : i32
            } else {
              %4 = arith.addi %0, %b : i32
              scf.yield %4 : i32
            }
            %5 = "test.op"(%2) : (i32) -> i32
            symref.update @a = %5 : i32
            scf.if %1 {
              "test.op"(%5) : (i32) -> ()
            } else {
            }
            func.return %5 : i32
          }
        }""")


def test_lower_calls_without_one_result():
    program = """
    py.func @f ["int"] -> "None" {
    ^bb0(%x: !py.object):
      %0 = py.call "consume"(%x)
      %1 = py.call "pair"(%x, %x)
      %2, %3 = py.unpack %1 : !py.object, !py.object
      %4 = py.build_tuple(%3, %2)
      %5, %6 = py.unpack %4 : !py.object, !py.object
      %7 = py.call "Counter.start"()
      %8 = py.call "pair"(%5, %7)
      py.return
    }
    """
    assert lower(program) == dedent("""\
        builtin.module {
          func.func @f(%x: i32) {
            "test.op"(%x) : (i32) -> ()
            %0, %1 = "test.op"(%x, %x) : (i32, i32) -> (i32, i32)
            %2 = "test.op"() : () -> i32
            %3, %4 = "test.op"(%1, %2) : (i32, i32) -> (i32, i32)
            func.return
          }
        }""")


def test_lower_with_custom_constructors():
    locations: list[Location] = []

    def build_function(
        name: str, signature: FunctionType, body: Region, location: Location
    ):
        locations.append(location)
        return build_func(name, signature, body, location)

    def omit_return(values: Sequence[SSAValue], location: Location) -> None:
        locations.append(location)

    program = """
    py.func @f [] -> "None" {
      py.return loc("test.py":3:5)
    } loc("test.py":2:1)
    """
    assert lower(
        program,
        function_definition_constructor=build_function,
        return_constructor=omit_return,
    ) == dedent("""\
        builtin.module {
          func.func @f() {
          }
        }""")
    assert locations == [Location("test.py", 2, 1), Location("test.py", 3, 5)]


def lower_body(body: str) -> str:
    """Lower `body` in a function `f(x: int) -> int`."""
    return lower(f"""
    py.func @f ["int"] -> "int" {{
    ^bb0(%x: !py.object):
      {body}
    }}
    """)


@pytest.mark.parametrize(
    "body, message",
    [
        (
            '%0 = py.const "s" loc("test.py":2:3)',
            "Unsupported constant 's' of type 'str'.",
        ),
        (
            """
            %0 = py.compare "lt" %x %x
            %1 = py.binop "add" %x %0 loc("test.py":2:3)
            """,
            "Expected the same types for binary operation 'Add', but got i32 and i1.",
        ),
        (
            """
            %0 = py.compare "lt" %x %x
            %1 = py.binop "add" %0 %0 loc("test.py":2:3)
            """,
            "IR type 'i1' is not registered with a source type.",
        ),
        (
            '%0 = py.binop "sub" %x %x loc("test.py":2:3)',
            "Binary operation 'Sub' is not supported by type 'int' which does not "
            "overload '__sub__'.",
        ),
        (
            """
            %0 = py.compare "lt" %x %x
            %1 = py.compare "eq" %x %0 loc("test.py":2:3)
            """,
            "Expected the same types for comparison operator 'Eq', but got i32 and i1.",
        ),
        (
            '%0 = py.compare "gt" %x %x loc("test.py":2:3)',
            "Comparison operation 'Gt' is not supported by type 'integer_type' which "
            "does not overload '__gt__'.",
        ),
        (
            '%0 = py.binop "and" %x %x loc("test.py":2:3)',
            "Unexpected binary operation and.",
        ),
        (
            '%0 = py.compare "is" %x %x loc("test.py":2:3)',
            "Unsupported comparison operation 'is'.",
        ),
        (
            '%0 = py.call "missing"() loc("test.py":2:3)',
            "Function 'missing' is not defined in scope.",
        ),
        (
            '%0 = py.call "Missing.start"() loc("test.py":2:3)',
            "Class 'Missing' is not defined in scope.",
        ),
        (
            '%0 = py.call "Counter.missing"() loc("test.py":2:3)',
            "Method 'missing' is not defined on class 'Counter'.",
        ),
        (
            '%0 = py.call "unregistered"() loc("test.py":2:3)',
            "Function 'unregistered' is not registered.",
        ),
        (
            '%0 = py.call "Counter.stop"() loc("test.py":2:3)',
            "Classmethod 'Counter.stop' is not registered.",
        ),
        (
            """
            %0 = py.call "consume"(%x) loc("test.py":2:3)
            py.return %0
            """,
            "Expected an expression with exactly one result.",
        ),
        (
            """
            %0 = py.build_tuple(%x, %x) loc("test.py":2:3)
            %1 = py.call "effect"(%0)
            """,
            "Expected an expression with exactly one result.",
        ),
        (
            """
            %0 = py.call "pair"(%x, %x)
            %1, %2, %3 = py.unpack %0 : !py.object, !py.object, !py.object loc("test.py":2:3)
            """,
            "Expected 3 values to unpack, but got 2.",
        ),
        (
            '%0, %1 = py.unpack %x : !py.object, !py.object loc("test.py":2:3)',
            "Expected 2 values to unpack, but got 1.",
        ),
        (
            '%0 = py.bool %x loc("test.py":2:3)',
            "Expected a condition of type i1, but got i32.",
        ),
        (
            """
            %0 = py.compare "lt" %x %x
            symref.declare "y"
            symref.update @y = %x : !py.object
            symref.update @y = %0 : !py.object loc("test.py":2:3)
            """,
            "Cannot change the type of variable 'y'.",
        ),
        (
            """
            %0 = py.compare "lt" %x %x
            %1 = py.bool %0
            %2 = scf.if %1 -> (!py.object) {
              scf.yield %x : !py.object
            } else {
              %3 = py.compare "lt" %x %x
              scf.yield %3 : !py.object
            } loc("test.py":2:3)
            """,
            "Expected the same types for if expression, but got i32 and i1.",
        ),
    ],
)
def test_lowering_errors(body: str, message: str):
    with pytest.raises(CodeGenerationException) as error:
        lower_body(body)
    assert str(error.value) == (
        f'Code generation exception at "test.py", line 2 column 2: {message}'
    )


def test_unsupported_annotations():
    with pytest.raises(CodeGenerationException) as error:
        lower("""
        py.func @f ["str"] -> "None" {
        ^bb0(%x: !py.object loc("test.py":2:7)):
          py.return
        }
        """)
    assert str(error.value) == (
        'Code generation exception at "test.py", line 2 column 6: '
        "Unsupported function argument type: 'str'"
    )

    with pytest.raises(CodeGenerationException) as error:
        lower("""
        py.func @f [] -> "str" {
          py.return
        } loc("test.py":2:1)
        """)
    assert str(error.value) == (
        'Code generation exception at "test.py", line 2 column 0: '
        "Unsupported function return type: 'str'"
    )


def test_error_without_location():
    with pytest.raises(CodeGenerationException) as error:
        lower_body('%0 = py.const "s"')
    assert str(error.value) == (
        'Code generation exception at "test.py", line 0 column 0: '
        "Unsupported constant 's' of type 'str'."
    )
