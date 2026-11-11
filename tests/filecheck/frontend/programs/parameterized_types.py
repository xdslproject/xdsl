# RUN: python %s | filecheck %s

import pytest

from xdsl.dialects import builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException

ctx = PyASTContext()
pair_type = builtin.TupleType((builtin.i32, builtin.i1))
ctx.register_type(tuple[int, bool], pair_type)
ctx.register_type(tuple[bool, int], builtin.TupleType((builtin.i1, builtin.i32)))


@ctx.parse_program
def identity(value: tuple[int, bool]) -> tuple[int, bool]:
    return value


print(identity.module)
# CHECK:      func.func @identity(%value: tuple<i32, i1>) -> tuple<i32, i1> {
# CHECK-NEXT:   func.return %value : tuple<i32, i1>
# CHECK-NEXT: }

assert ctx.type_registry.get_annotation(pair_type) is tuple
with pytest.raises(FrontendProgramException, match="Cannot re-register type name"):
    ctx.register_type(tuple[int, bool], pair_type)
with pytest.raises(
    FrontendProgramException, match="Cannot register multiple source types"
):
    ctx.register_type(list[int], pair_type)
