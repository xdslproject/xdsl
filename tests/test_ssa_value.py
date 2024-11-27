import pytest

from xdsl.dialects.builtin import StringAttr, i32
from xdsl.ir import Block, BlockArgument, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def


@irdl_op_definition
class TwoResultOp(IRDLOperation):
    name = "test.tworesults"

    res1 = result_def(StringAttr)
    res2 = result_def(StringAttr)


def test_var_mixed_builder():
    op = TwoResultOp.build(result_types=[StringAttr("0"), StringAttr("2")])

    with pytest.raises(ValueError):
        _ = SSAValue.get(op)


@pytest.mark.parametrize(
    "name",
    [
        "test",
        "-2",
        "test123",
        "kebab-case-name",
        None,
    ],
)
def test_ssa_value_name_hints(name: str | None):
    r"""
    As per the MLIR language reference, legal SSA value names must conform to
        ([0-9]+|([A-Za-z_$.-][\w$.-]*))

    https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords

    xDSL SSA value name hints are a refinement of these rules.
    We only accept non-numeric name hints, because the printer will
    generate its own numeric names.

    This test tests valid name hints:
    """
    val = BlockArgument(i32, Block(), 0)

    val.name_hint = name
    assert val.name_hint == name


@pytest.mark.parametrize("name", ["&", "#", "%2", '"', "::", "42"])
def test_invalid_ssa_vals(name: str):
    """
    This test tests invalid name hints that raise an error, because
    they don't conform to the rules of how SSA value names should be
    structured.
    """
    val = BlockArgument(i32, Block(), 0)
    with pytest.raises(ValueError):
        val.name_hint = name
