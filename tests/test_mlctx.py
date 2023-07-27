import pytest

from xdsl.dialects.builtin import UnregisteredAttr, UnregisteredOp
from xdsl.ir import MLContext, Operation, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition


class DummyOp(Operation):
    name = "dummy"


class DummyOp2(Operation):
    name = "dummy2"


@irdl_attr_definition
class DummyAttr(ParametrizedAttribute):
    name = "dummy_attr"


@irdl_attr_definition
class DummyAttr2(ParametrizedAttribute):
    name = "dummy_attr2"


def test_get_op():
    """Test `get_op` and `get_optional_op` methods."""
    ctx = MLContext()
    ctx.register_op(DummyOp)

    assert ctx.get_op("dummy") == DummyOp
    with pytest.raises(Exception):
        _ = ctx.get_op("dummy2")

    assert ctx.get_optional_op("dummy") == DummyOp
    assert ctx.get_optional_op("dummy2") is None


def test_get_op_unregistered():
    """
    Test `get_op` and `get_optional_op`
    methods with the `allow_unregistered` flag.
    """
    ctx = MLContext(allow_unregistered=True)
    ctx.register_op(DummyOp)

    assert ctx.get_optional_op("dummy") == DummyOp
    op = ctx.get_optional_op("dummy2")
    assert op is not None
    assert issubclass(op, UnregisteredOp)

    assert ctx.get_op("dummy") == DummyOp
    assert issubclass(ctx.get_op("dummy2"), UnregisteredOp)


def test_get_attr():
    """Test `get_attr` and `get_optional_attr` methods."""
    ctx = MLContext()
    ctx.register_attr(DummyAttr)

    assert ctx.get_attr("dummy_attr") == DummyAttr
    with pytest.raises(Exception):
        _ = ctx.get_attr("dummy_attr2")

    assert ctx.get_optional_attr("dummy_attr") == DummyAttr
    assert ctx.get_optional_attr("dummy_attr2") is None


@pytest.mark.parametrize("is_type", [True, False])
def test_get_attr_unregistered(is_type: bool):
    """
    Test `get_attr` and `get_optional_attr`
    methods with the `allow_unregistered` flag.
    """
    ctx = MLContext(allow_unregistered=True)
    ctx.register_attr(DummyAttr)

    assert (
        ctx.get_optional_attr("dummy_attr", create_unregistered_as_type=is_type)
        == DummyAttr
    )
    attr = ctx.get_optional_attr("dummy_attr2")
    assert attr is not None
    assert issubclass(attr, UnregisteredAttr)
    if is_type:
        assert issubclass(attr, TypeAttribute)

    assert ctx.get_attr("dummy_attr", create_unregistered_as_type=is_type) == DummyAttr
    assert issubclass(
        ctx.get_attr("dummy_attr2", create_unregistered_as_type=is_type),
        UnregisteredAttr,
    )
    if is_type:
        assert issubclass(attr, TypeAttribute)
