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


@irdl_op_definition
class TwoResultOp(Operation):
    name: str = "test.tworesults"

    res1: Annotated[OpResult, StringAttr]
    res2: Annotated[OpResult, StringAttr]


def test_var_mixed_builder():
    op = TwoResultOp.build(result_types=[StringAttr("0"), StringAttr("2")])
    b = OpResult(i32, [], [])

    with pytest.raises(ValueError):
        _ = b.get(op)


@pytest.mark.parametrize("name",
                         ["a", "test", "-2", "test_123", "kebab-case-name"])
def test_ssa_value_name_hints(name):
    r"""
    As per the MLIR language reference, legal SSA value names must conform to
        ([0-9]+|([A-Za-z_$.-][\w$.-]*))

    https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords

    xDSL SSA value name hints are a refinement of these rules.
    We only accept non-numeric name hints, because the printer will
    generate its own numeric names.

    The next three tests will test this behaviour.

    This test tests valid name hints:
    """
    val = BlockArgument(i32, Block(), 0)

    val.name = name
    assert val.name == name


@pytest.mark.parametrize("name", ['&', '#', '%2', '"', '::'])
def test_invalid_ssa_vals(name):
    """
    This test tests invalid name hints that raise an error, because
    they don't conform to the rules of how SSA value names should be
    structured.
    """
    val = BlockArgument(i32, Block(), 0)
    with pytest.raises(ValueError):
        val.name = name


@pytest.mark.parametrize("name", ['2', '500', '42', '69', None])
def test_discarded_ssa_vals(name: str | None):
    """
    This test tests ssa value name hints that should be discarded.
    """
    val = BlockArgument(i32, Block(), 0)

    val.name = name

    assert val.name is None
