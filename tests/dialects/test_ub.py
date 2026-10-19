from xdsl.dialects.builtin import VectorType, i32, i64
from xdsl.dialects.ub import PoisonAttr, PoisonOp


def test_poison_default_value():
    """Constructing without a value defaults to the empty `#ub.poison` attr."""
    op = PoisonOp(i32)
    assert op.result.type == i32
    assert op.value == PoisonAttr()


def test_poison_explicit_value():
    """An explicit poison attribute is stored verbatim on the `value` prop."""
    value = PoisonAttr()
    op = PoisonOp(VectorType(i64, [4]), value)
    assert op.result.type == VectorType(i64, [4])
    assert op.value is value
