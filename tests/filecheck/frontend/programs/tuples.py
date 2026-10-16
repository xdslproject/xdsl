# RUN: python %s | filecheck %s

from collections.abc import Sequence

import pytest

from xdsl.dialects import arith, builtin, test
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import SSAValue, TypeAttribute
from xdsl.utils.hints import isa


def build_tuple(elements: Sequence[SSAValue]) -> test.TestOp:
    types = tuple(element.type for element in elements)
    assert isa(types, tuple[TypeAttribute, ...])
    return test.TestOp(elements, [builtin.TupleType(types)])


ctx = PyASTContext()
ctx.register_type(int, builtin.i32)
ctx.register_type(tuple[int, int], builtin.TupleType((builtin.i32, builtin.i32)))
ctx.register_literal(int, lambda value: arith.ConstantOp.from_int_and_width(value, 32))
ctx.register_tuple(build_tuple)


@ctx.parse_program
def pair(x: int) -> tuple[int, int]:
    values = (x, 7)
    return values


assert pair(3) == (3, 7)
print(pair.module)
# CHECK:      func.func @pair(%x: i32) -> tuple<i32, i32> {
# CHECK-NEXT:   %0 = arith.constant 7 : i32
# CHECK-NEXT:   %1 = "test.op"(%x, %0) : (i32, i32) -> tuple<i32, i32>
# CHECK-NEXT:   func.return %1 : tuple<i32, i32>
# CHECK-NEXT: }


for result_types in ((), (builtin.i32, builtin.i32)):
    ctx.register_tuple(lambda elements: test.TestOp(elements, result_types))

    @ctx.parse_program
    def invalid() -> None:
        _ = ()

    with pytest.raises(
        CodeGenerationException, match="tuple constructor with exactly one result"
    ):
        invalid.module


unregistered = PyASTContext()


@unregistered.parse_program
def missing_constructor() -> None:
    _ = ()


with pytest.raises(
    CodeGenerationException, match="Tuple construction is not registered"
):
    missing_constructor.module
