import pytest

from xdsl.context import Context
from xdsl.dialects import arith, test
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, i32
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.dialects.py import Py
from xdsl.dialects.symref import Symref
from xdsl.dialects.test import Test
from xdsl.frontend.pyast.lowering import PyLowering
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.ir import SSAValue
from xdsl.parser import Parser


def effect(value: int) -> int: ...
def consume(value: int) -> None: ...
def pair(lhs: int, rhs: int) -> tuple[int, int]: ...


class Counter:
    @classmethod
    def start(cls) -> int: ...


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


def lower(program: str) -> str:
    ctx = Context()
    for dialect in (Arith, Builtin, Cf, Func, Py, Symref, Test):
        ctx.load_dialect(dialect)
    module = Parser(ctx, program).parse_module()

    type_converter = TypeConverter(globals())
    type_converter.type_registry.insert(int, i32)
    type_converter.function_registry.insert(int.__add__, arith.AddiOp)
    type_converter.function_registry.insert(int.__lt__, build_less_than)
    type_converter.function_registry.insert(effect, build_effect)
    type_converter.function_registry.insert(consume, build_consume)
    type_converter.function_registry.insert(pair, build_pair)
    type_converter.function_registry.insert(Counter.start, build_start)

    PyLowering(type_converter, "test.py").lower(module)
    return str(module)


def test_lower_function():
    program = """
    py.func @f(%a : !py.object<"int">, %b : !py.object<"int">) -> !py.object<"int"> {
      symref.declare "a"
      symref.update @a = %a : !py.object<"int">
      symref.declare "b"
      symref.update @b = %b : !py.object<"int">
      %0 = symref.fetch @a : !py.object
      %1 = symref.fetch @b : !py.object
      %2 = py.compare lt, %0, %1
      "cf.assert"(%2) <{msg = ""}> : (!py.object) -> ()
      %3 = py.binop add, %0, %1
      %4 = py.call "effect"(%3) keywords ["value"]
      symref.update @a = %4 : !py.object
      %5 = py.call "consume"(%4)
      %6 = py.cast %4 : !py.object to !py.object<"int">
      py.return %6 : !py.object<"int">
    }
    """
    assert (
        lower(program)
        == """\
builtin.module {
  func.func @f(%a: i32, %b: i32) -> i32 {
    symref.declare "a"
    symref.update @a = %a : i32
    symref.declare "b"
    symref.update @b = %b : i32
    %0 = symref.fetch @a : i32
    %1 = symref.fetch @b : i32
    %2 = arith.cmpi slt, %0, %1 : i32
    cf.assert %2, ""
    %3 = arith.addi %0, %1 : i32
    %4 = "test.op"(%3) : (i32) -> i32
    symref.update @a = %4 : i32
    "test.op"(%4) : (i32) -> ()
    func.return %4 : i32
  }
}"""
    )


def test_lower_calls_without_one_result():
    program = """
    py.func @f(%x : !py.object<"int">) -> !py.object<"None"> {
      symref.declare "x"
      symref.update @x = %x : !py.object<"int">
      %0 = symref.fetch @x : !py.object
      %1 = py.call "consume"(%0)
      %2 = py.call "pair"(%0, %0)
      %3 = py.call "Counter.start"()
      %none = py.const None
      %result = py.cast %none : !py.object to !py.object<"None">
      py.return %result : !py.object<"None">
    }
    """
    assert (
        lower(program)
        == """\
builtin.module {
  func.func @f(%x: i32) {
    symref.declare "x"
    symref.update @x = %x : i32
    %0 = symref.fetch @x : i32
    "test.op"(%0) : (i32) -> ()
    %1, %2 = "test.op"(%0, %0) : (i32, i32) -> (i32, i32)
    %3 = "test.op"() : () -> i32
    func.return
  }
}"""
    )


def lower_body(body: str) -> str:
    """Lower `body` in a function `f(x: int) -> int`, where `%x` is `x`."""
    return lower(f"""
    py.func @f(%arg : !py.object<"int">) -> !py.object<"int"> {{
      symref.declare "x"
      symref.update @x = %arg : !py.object<"int">
      %x = symref.fetch @x : !py.object
      {body}
    }}
    """)


def test_lowering_error():
    with pytest.raises(CodeGenerationException) as error:
        lower_body("""
            %0 = py.compare lt, %x, %x
            %1 = py.binop add, %0, %0 loc("test.py":2:3)
            """)
    assert str(error.value) == (
        'Code generation exception at "test.py", line 2 column 2: '
        "IR type 'i1' is not registered with a source type."
    )


def test_error_without_location():
    with pytest.raises(CodeGenerationException) as error:
        lower_body('%0 = py.call "missing"()')
    assert str(error.value) == (
        'Code generation exception at "test.py", line 0 column 0: '
        "Function 'missing' is not defined in scope."
    )
