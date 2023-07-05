import pytest

from xdsl.dialects.affine import For, Yield
from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr, IntegerType
from xdsl.ir import Attribute, Region, Block
from xdsl.ir.affine.affine_expr import AffineExpr


def test_simple_for():
    f = For.from_region([], [], 0, 5, Region())
    assert f.lower_bound.data.results == [AffineExpr.constant(0)]
    assert f.upper_bound.data.results == [AffineExpr.constant(5)]


def test_for_mismatch_operands_results_counts():
    attributes: dict[str, Attribute] = {
        "lower_bound": AffineMapAttr.constant_map(0),
        "upper_bound": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    f = For.create(
        operands=[],
        regions=[Region()],
        attributes=attributes,
        result_types=[IndexType()],
    )
    with pytest.raises(Exception) as e:
        f.verify()
    assert e.value.args[0] == "Expected the same amount of operands and results"


def test_for_mismatch_operands_results_types():
    attributes: dict[str, Attribute] = {
        "lower_bound": AffineMapAttr.constant_map(0),
        "upper_bound": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    b = Block(arg_types=(IntegerType(32),))
    inp = b.args[0]
    f = For.create(
        operands=[inp],
        regions=[Region()],
        attributes=attributes,
        result_types=[IndexType()],
    )
    with pytest.raises(Exception) as e:
        f.verify()
    assert (
        e.value.args[0]
        == "Expected all operands and result pairs to have matching types"
    )


def test_for_mismatch_blockargs():
    attributes: dict[str, Attribute] = {
        "lower_bound": AffineMapAttr.constant_map(0),
        "upper_bound": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    b = Block(arg_types=(IndexType(),))
    inp = b.args[0]
    f = For.create(
        operands=[inp],
        regions=[Region(Block([Yield.get()]))],
        attributes=attributes,
        result_types=[IndexType()],
    )
    with pytest.raises(Exception) as e:
        f.verify()
    assert (
        e.value.args[0]
        == "Expected BlockArguments to have the same types as the operands"
    )


def test_yield():
    yield_ = Yield.get()
    assert yield_.arguments == ()
