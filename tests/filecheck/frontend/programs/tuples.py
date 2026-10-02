# RUN: python %s | filecheck %s

from collections.abc import Sequence

import pytest

from xdsl.dialects import arith, builtin, test
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import SSAValue, TypeAttribute
from xdsl.utils.hints import isa


def consume(value: object) -> None: ...


def build_tuple(elements: Sequence[SSAValue]) -> test.TestOp:
    types = tuple(element.type for element in elements)
    assert isa(types, tuple[TypeAttribute, ...])
    return test.TestOp(elements, [builtin.TupleType(types)])


def build_consume(value: SSAValue) -> test.TestOp:
    return test.TestOp([value])


ctx = PyASTContext()
ctx.register_type(int, builtin.i32)
ctx.register_literal(int, lambda value: arith.ConstantOp.from_int_and_width(value, 32))
ctx.register_function(consume, build_consume)
ctx.register_tuple(build_tuple)


# Empty and nested tuples, with both a runtime value and a literal element.
@ctx.parse_program
def tuples(x: int) -> None:
    values = ((), (x, 1))
    consume(values)


print(tuples.module)
# CHECK:      func.func @tuples(%x: i32) {
# CHECK-NEXT:   %0 = "test.op"() : () -> tuple<>
# CHECK-NEXT:   %1 = arith.constant 1 : i32
# CHECK-NEXT:   %2 = "test.op"(%x, %1) : (i32, i32) -> tuple<i32, i32>
# CHECK-NEXT:   %3 = "test.op"(%0, %2) : (tuple<>, tuple<i32, i32>) -> tuple<tuple<>, tuple<i32, i32>>
# CHECK-NEXT:   "test.op"(%3) : (tuple<tuple<>, tuple<i32, i32>>) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }


@ctx.parse_program
def missing_element_result(x: int) -> None:
    consume((consume(x),))


with pytest.raises(CodeGenerationException, match="expression with exactly one result"):
    missing_element_result.module


for result_types in ((), (builtin.i32, builtin.i32)):
    ctx.register_tuple(lambda elements: test.TestOp(elements, result_types))

    @ctx.parse_program
    def invalid_constructor() -> None:
        consume(())

    with pytest.raises(
        CodeGenerationException, match="tuple constructor with exactly one result"
    ):
        invalid_constructor.module


unregistered = PyASTContext()


@unregistered.parse_program
def missing_constructor() -> None:
    _ = ()


with pytest.raises(
    CodeGenerationException, match="Tuple construction is not registered"
):
    missing_constructor.module
