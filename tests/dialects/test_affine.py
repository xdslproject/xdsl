import pytest

from xdsl.dialects.affine import For, Yield
from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr, IntegerType
from xdsl.ir import Attribute, Block, Region
from xdsl.ir.affine import AffineExpr
from xdsl.utils.exceptions import VerifyException


def test_simple_for():
    f = For.from_region([], [], 0, 5, Region())
    assert f.lower_bound.data.results == (AffineExpr.constant(0),)
    assert f.upper_bound.data.results == (AffineExpr.constant(5),)


def test_for_mismatch_operands_results_counts():
    properties: dict[str, Attribute] = {
        "lower_bound": AffineMapAttr.constant_map(0),
        "upper_bound": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    f = For.create(
        operands=[],
        regions=[Region()],
        properties=properties,
        result_types=[IndexType()],
    )
    with pytest.raises(
        VerifyException,
        match="Expected as many operands as results, lower bound args and upper bound args.",
    ):
        f.verify()


def test_for_mismatch_operands_results_types():
    properties: dict[str, Attribute] = {
        "lower_bound": AffineMapAttr.constant_map(0),
        "upper_bound": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    b = Block(arg_types=(IntegerType(32),))
    inp = b.args[0]
    f = For.create(
        operands=[inp],
        regions=[Region()],
        properties=properties,
        result_types=[IndexType()],
    )
    with pytest.raises(
        VerifyException,
        match="Expected all operands and result pairs to have matching types",
    ):
        f.verify()


def test_for_mismatch_blockargs():
    properties: dict[str, Attribute] = {
        "lower_bound": AffineMapAttr.constant_map(0),
        "upper_bound": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    b = Block(arg_types=(IndexType(),))
    inp = b.args[0]
    f = For.create(
        operands=[inp],
        regions=[Region(Block([Yield.get()]))],
        properties=properties,
        result_types=[IndexType()],
    )
    with pytest.raises(
        VerifyException,
        match="Expected BlockArguments to have the same types as the operands",
    ):
        f.verify()


def test_yield():
    yield_ = Yield.get()
    assert yield_.arguments == ()
