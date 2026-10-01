# RUN: python %s | filecheck %s

from xdsl.dialects import arith, builtin, test
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import SSAValue


def add(operand1: int, operand2: int) -> int: ...


def pair(operand1: int, operand2: int) -> tuple[int, bool]: ...


def consume(value: int) -> None: ...


def build_consume(value: SSAValue) -> test.TestOp:
    return test.TestOp([value])


ctx = PyASTContext()
ctx.register_type(int, builtin.i32)
ctx.register_function(add, arith.AddiOp)
ctx.register_function(pair, arith.AddUIExtendedOp)
ctx.register_function(consume, build_consume)


@ctx.parse_program
def nested(x: int, y: int) -> int:
    return add(add(x, y), operand2=add(y, x))


print(nested.module)
# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @nested(%x: i32, %y: i32) -> i32 {
# CHECK-NEXT:     %0 = arith.addi %x, %y : i32
# CHECK-NEXT:     %1 = arith.addi %y, %x : i32
# CHECK-NEXT:     %2 = arith.addi %0, %1 : i32
# CHECK-NEXT:     func.return %2 : i32
# CHECK-NEXT:   }
# CHECK-NEXT: }


@ctx.parse_program
def return_zero_results(x: int) -> int:
    return consume(x)  # pyright: ignore[reportReturnType]


@ctx.parse_program
def return_multiple_results(x: int, y: int) -> int:
    return pair(x, y)  # pyright: ignore[reportReturnType]


@ctx.parse_program
def argument_zero_results(x: int, y: int) -> int:
    return add(consume(x), y)  # pyright: ignore[reportArgumentType]


@ctx.parse_program
def keyword_multiple_results(x: int, y: int) -> int:
    return add(x, operand2=pair(x, y))  # pyright: ignore[reportArgumentType]


@ctx.parse_program
def unpack_keywords(x: int, y: int) -> int:
    return add(**{"operand1": x, "operand2": y})


for program in (
    return_zero_results,
    return_multiple_results,
    argument_zero_results,
    keyword_multiple_results,
    unpack_keywords,
):
    try:
        program.module
    except CodeGenerationException as error:
        print(f"{program.name}: {error.msg}")
    else:
        raise AssertionError("expected a frontend diagnostic")
# CHECK-NEXT: return_zero_results: Expected an expression with exactly one result.
# CHECK-NEXT: return_multiple_results: Expected an expression with exactly one result.
# CHECK-NEXT: argument_zero_results: Expected an expression with exactly one result.
# CHECK-NEXT: keyword_multiple_results: Expected an expression with exactly one result.
# CHECK-NEXT: unpack_keywords: Unpacking keyword arguments is not supported.
