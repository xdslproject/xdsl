import pytest

from xdsl.dialects.builtin import i32, StringAttr
from xdsl.dialects.arith import Constant

from xdsl.ir import Block, Operation, OpResult
from xdsl.irdl import irdl_op_definition, ResultDef


def test_ssa():
    a = OpResult(i32, [], [])
    c = Constant.from_int_and_width(1, i32)

    with pytest.raises(TypeError):
        _ = a.get([c])

    b0 = Block.from_ops([c])
    with pytest.raises(TypeError):
        _ = a.get(b0)


@irdl_op_definition
class TwoResultOp(Operation):
    name: str = "test.tworesults"

    res1 = ResultDef(StringAttr)
    res2 = ResultDef(StringAttr)


def test_var_mixed_builder():
    op = TwoResultOp.build(result_types=[0, 2])
    b = OpResult(i32, [], [])

    with pytest.raises(ValueError):
        _ = b.get(op)
