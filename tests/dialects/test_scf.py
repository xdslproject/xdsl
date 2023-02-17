from pytest import raises

from xdsl.dialects.arith import Constant, IndexType
from xdsl.dialects.builtin import Region
from xdsl.dialects.cf import Block
from xdsl.dialects.scf import For


def test_for():
    lb = Constant.from_int_and_width(0, IndexType())
    ub = Constant.from_int_and_width(42, IndexType())
    step = Constant.from_int_and_width(3, IndexType())
    carried = Constant.from_int_and_width(1, IndexType())
    bodyblock = Block.from_arg_types([IndexType()])
    body = Region.from_block_list([bodyblock])
    f = For.get(lb, ub, step, [carried], body)

    assert f.lb is lb.result
    assert f.ub is ub.result
    assert f.step is step.result
    assert f.iter_args == tuple([carried.result])
    assert f.body is body

    assert len(f.results) == 1
    assert f.results[0].typ == carried.result.typ
    assert f.operands == (lb.result, ub.result, step.result, carried.result)
    assert f.regions == [body]
    assert f.attributes == {}
