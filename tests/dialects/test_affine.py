import pytest

from xdsl.context import Context
from xdsl.dialects.affine import Affine, ForOp, YieldOp
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr, IntegerType
from xdsl.dialects.tensor import Tensor
from xdsl.ir import Attribute, Block, Region
from xdsl.ir.affine import AffineExpr
from xdsl.parser import Parser
from xdsl.utils.exceptions import VerifyException


def test_simple_for():
    f = ForOp.from_region([], [], [], [], 0, 5, Region())
    assert f.lowerBoundMap.data.results == (AffineExpr.constant(0),)
    assert f.upperBoundMap.data.results == (AffineExpr.constant(5),)


def test_for_mismatch_operands_results_counts():
    properties: dict[str, Attribute] = {
        "lowerBoundMap": AffineMapAttr.constant_map(0),
        "upperBoundMap": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    f = ForOp.build(
        operands=[[], [], []],
        regions=[Region()],
        properties=properties,
        result_types=[IndexType()],
    )
    with pytest.raises(
        VerifyException,
        match="Expected as many init operands as results.",
    ):
        f.verify()


def test_for_mismatch_operands_results_types():
    properties: dict[str, Attribute] = {
        "lowerBoundMap": AffineMapAttr.constant_map(0),
        "upperBoundMap": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    b = Block(arg_types=(IntegerType(32),))
    inp = b.args[0]
    f = ForOp.build(
        operands=[[], [], inp],
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
        "lowerBoundMap": AffineMapAttr.constant_map(0),
        "upperBoundMap": AffineMapAttr.constant_map(5),
        "step": IntegerAttr.from_index_int_value(1),
    }
    b = Block(arg_types=(IndexType(),))
    inp = b.args[0]
    f = ForOp.build(
        operands=[[], [], inp],
        regions=[Region(Block([YieldOp.get()]))],
        properties=properties,
        result_types=[IndexType()],
    )
    with pytest.raises(
        VerifyException,
        match="Expected BlockArguments to have the same types as the operands",
    ):
        f.verify()


def test_yield():
    yield_ = YieldOp.get()
    assert yield_.arguments == ()


def test_affine_apply_map():
    MODULE_CTX = """
#map = affine_map<()[s0] -> (s0 * 4)>
%c0 = arith.constant 2 : index
%0 = affine.apply #map()[%c0]
    """
    ctx = Context()
    ctx.load_dialect(Tensor)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Affine)

    module_op = Parser(ctx, MODULE_CTX).parse_module()

    print(module_op)

    module_op.verify()
