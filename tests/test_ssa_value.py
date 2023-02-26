import pytest

from typing import Annotated

from xdsl.dialects.builtin import i32, StringAttr
from xdsl.dialects.arith import Constant

from xdsl.ir import Block, Operation, OpResult, BlockArgument
from xdsl.irdl import irdl_op_definition


def test_ssa():
    a = OpResult(i32, [], [])
    c = Constant.from_int_and_width(1, i32)

    with pytest.raises(TypeError):
        _ = a.get([c])

    b0 = Block.from_ops([c])
    with pytest.raises(TypeError):
        _ = a.get(b0)


class TwoResultOp(Operation):
    name: str = "test.tworesults"

    res1: Annotated[OpResult, StringAttr]
    res2: Annotated[OpResult, StringAttr]


def test_var_mixed_builder():
    op = TwoResultOp.build(result_types=[StringAttr("0"), StringAttr("2")])
    b = OpResult(i32, [], [])

    with pytest.raises(ValueError):
        _ = b.get(op)


@pytest.mark.parametrize("name,result", [
    ('a', 'a'),
    ('test', 'test'),
    ('test1', None),
    ('1', None),
])
def test_ssa_value_name_hints(name, result):
    """
    The rewriter assumes, that ssa value name hints (their .name field) does not end in
    a numeric value. If it does, it will generate broken rewrites that potentially assign
    twice to an SSA value.

    Therefore, the SSAValue class prevents the setting of names ending in a number.
    """
    val = BlockArgument(i32, Block(), 0)

    val.name = name
    assert val.name == result
