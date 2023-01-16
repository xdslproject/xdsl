import pytest

from xdsl.dialects.affine import For, Yield
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Region, OpResult, Block


def test_simple_for():
    f = For.from_callable([], 0, 5, lambda x: [])
    assert f.lower_bound.value.data == 0
    assert f.upper_bound.value.data == 5


def test_for_mismatch_operands_results_counts():
    attributes = {
        "lower_bound": IntegerAttr.from_index_int_value(0),
        "upper_bound": IntegerAttr.from_index_int_value(5),
        "step": IntegerAttr.from_index_int_value(1)
    }
    f = For.create(operands=[],
                   regions=[Region.from_block_list([])],
                   attributes=attributes,
                   result_types=[IndexType])
    with pytest.raises(Exception) as e:
        f.verify()
    assert e.value.args[
        0] == "Expected the same amount of operands and results"


def test_for_mismatch_operands_results_types():
    attributes = {
        "lower_bound": IntegerAttr.from_index_int_value(0),
        "upper_bound": IntegerAttr.from_index_int_value(5),
        "step": IntegerAttr.from_index_int_value(1)
    }
    inp = OpResult(IntegerType.from_width(32), [], [])
    f = For.create(operands=[inp],
                   regions=[Region.from_block_list([])],
                   attributes=attributes,
                   result_types=[IndexType])
    with pytest.raises(Exception) as e:
        f.verify()
    assert e.value.args[
        0] == "Expected all operands and result pairs to have matching types"


def test_for_mismatch_blockargs():
    attributes = {
        "lower_bound": IntegerAttr.from_index_int_value(0),
        "upper_bound": IntegerAttr.from_index_int_value(5),
        "step": IntegerAttr.from_index_int_value(1)
    }
    inp = OpResult(IndexType, [], [])
    f = For.create(operands=[inp],
                   regions=[
                       Region.from_block_list(
                           [Block.from_callable([], lambda: [])])
                   ],
                   attributes=attributes,
                   result_types=[IndexType])
    with pytest.raises(Exception) as e:
        f.verify()
    assert e.value.args[
        0] == "Expected BlockArguments to have the same types as the operands"


def test_yield():
    yield_ = Yield.get()
    assert yield_.arguments == []
