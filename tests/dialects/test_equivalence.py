import pytest

from xdsl.dialects import arith, equivalence, test
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.traits import ConstantLike
from xdsl.utils.exceptions import DiagnosticException


def test_const_class_construction():
    value = IntegerAttr(42, 64)
    constant_op = arith.ConstantOp(value)

    const_class = equivalence.ConstantClassOp(constant_op.result)
    trait = const_class.get_trait(ConstantLike)
    assert trait is not None
    assert ConstantLike.get_constant_value(constant_op.result) == value

    non_constant_op = test.TestOp(result_types=(i32,))
    with pytest.raises(
        DiagnosticException,
        match="The argument of a ConstantClass must be a `ConstantLike` operation implementing `HasFolderInterface`.",
    ):
        equivalence.ConstantClassOp(non_constant_op.results[0])
