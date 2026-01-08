import pytest

from xdsl.dialects import arith, equivalence, test
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.traits import ConstantLike
from xdsl.utils.exceptions import DiagnosticException


def test_const_class_construction():
    constant_op = arith.ConstantOp(IntegerAttr.from_int_and_width(42, 64))

    const_class = equivalence.ConstantClassOp(constant_op.result)
    trait = const_class.get_trait(ConstantLike)
    assert trait is not None
    assert trait.get_constant_value(const_class) == IntegerAttr.from_int_and_width(
        42, 64
    )

    non_constant_op = test.TestOp(result_types=(i32,))
    with pytest.raises(
        DiagnosticException,
        match="The argument of a ConstantClass must be a constant-like operation.",
    ):
        equivalence.ConstantClassOp(non_constant_op.results[0])
