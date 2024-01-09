import pytest

from xdsl.dialects.builtin import UnregisteredAttr, UnregisteredOp
from xdsl.ir import Dialect, MLContext, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import IRDLOperation, irdl_attr_definition, irdl_op_definition


@irdl_op_definition
class DummyOp(IRDLOperation):
    name = "test.dummy"


@irdl_op_definition
class DummyOp2(IRDLOperation):
    name = "test.dummy2"


@irdl_attr_definition
class DummyAttr(ParametrizedAttribute):
    name = "test.dummy_attr"


@irdl_attr_definition
class DummyAttr2(ParametrizedAttribute):
    name = "test.dummy_attr2"


testDialect = Dialect("test", [DummyOp], [DummyAttr])
testDialect2 = Dialect("test", [DummyOp2], [DummyAttr2])


def test_get_op():
    """Test `get_op` and `get_optional_op` methods."""
    ctx = MLContext()
    ctx.load_op(DummyOp)

    assert ctx.get_op("test.dummy") == DummyOp
    with pytest.raises(Exception):
        _ = ctx.get_op("test.dummy2")

    assert ctx.get_optional_op("test.dummy") == DummyOp
    assert ctx.get_optional_op("test.dummy2") is None


def test_get_op_unregistered():
    """
    Test `get_op` and `get_optional_op`
    methods with the `allow_unregistered` flag.
    """
    ctx = MLContext(allow_unregistered=True)
    ctx.load_op(DummyOp)

    assert ctx.get_optional_op("test.dummy") == DummyOp
    op = ctx.get_optional_op("test.dummy2")
    assert op is not None
    assert issubclass(op, UnregisteredOp)

    assert ctx.get_op("test.dummy") == DummyOp
    assert issubclass(ctx.get_op("test.dummy2"), UnregisteredOp)


def test_get_attr():
    """Test `get_attr` and `get_optional_attr` methods."""
    ctx = MLContext()
    ctx.load_attr(DummyAttr)

    assert ctx.get_attr("test.dummy_attr") == DummyAttr
    with pytest.raises(Exception):
        _ = ctx.get_attr("test.dummy_attr2")

    assert ctx.get_optional_attr("test.dummy_attr") == DummyAttr
    assert ctx.get_optional_attr("test.dummy_attr2") is None


@pytest.mark.parametrize("is_type", [True, False])
def test_get_attr_unregistered(is_type: bool):
    """
    Test `get_attr` and `get_optional_attr`
    methods with the `allow_unregistered` flag.
    """
    ctx = MLContext(allow_unregistered=True)
    ctx.load_attr(DummyAttr)

    assert (
        ctx.get_optional_attr("test.dummy_attr", create_unregistered_as_type=is_type)
        == DummyAttr
    )
    attr = ctx.get_optional_attr("test.dummy_attr2")
    assert attr is not None
    assert issubclass(attr, UnregisteredAttr)
    if is_type:
        assert issubclass(attr, TypeAttribute)

    assert (
        ctx.get_attr("test.dummy_attr", create_unregistered_as_type=is_type)
        == DummyAttr
    )
    assert issubclass(
        ctx.get_attr("test.dummy_attr2", create_unregistered_as_type=is_type),
        UnregisteredAttr,
    )
    if is_type:
        assert issubclass(attr, TypeAttribute)


def test_clone_function():
    ctx = MLContext()
    ctx.load_attr(DummyAttr)
    ctx.load_op(DummyOp)

    copy = ctx.clone()

    assert ctx == copy
    copy.load_op(DummyOp2)
    assert ctx != copy

    copy = ctx.clone()

    assert ctx == copy
    copy.load_attr(DummyAttr2)
    assert ctx != copy


def test_register_dialect_get_op_attr():
    ctx = MLContext()
    ctx.register_dialect("test", lambda: testDialect)
    op = ctx.get_op("test.dummy")
    assert op == DummyOp
    attr = ctx.get_attr("test.dummy_attr")
    assert attr == DummyAttr
    assert "test" in ctx.registered_dialect_names
    assert list(ctx.loaded_dialects) == [testDialect]


def test_register_dialect_already_registered():
    ctx = MLContext()
    ctx.register_dialect("test", lambda: testDialect)
    with pytest.raises(ValueError, match="'test' dialect is already registered"):
        ctx.register_dialect("test", lambda: testDialect2)


def test_register_dialect_already_loaded():
    ctx = MLContext()
    ctx.load_dialect(testDialect)
    with pytest.raises(ValueError, match="'test' dialect is already registered"):
        ctx.register_dialect("test", lambda: testDialect2)


def test_load_registered_dialect():
    ctx = MLContext()
    ctx.register_dialect("test", lambda: testDialect)
    assert list(ctx.loaded_dialects) == []
    assert list(ctx.registered_dialect_names) == ["test"]
    ctx.load_registered_dialect("test")
    assert list(ctx.loaded_dialects) == [testDialect]
    assert list(ctx.registered_dialect_names) == ["test"]


def test_load_registered_dialect_not_registered():
    ctx = MLContext()
    with pytest.raises(ValueError, match="'test' dialect is not registered"):
        ctx.load_registered_dialect("test")


def test_load_dialect():
    ctx = MLContext()
    ctx.load_dialect(testDialect)
    assert list(ctx.loaded_dialects) == [testDialect]
    assert list(ctx.registered_dialect_names) == ["test"]


def test_load_dialect_already_loaded():
    ctx = MLContext()
    ctx.load_dialect(testDialect)
    with pytest.raises(ValueError, match="'test' dialect is already registered"):
        ctx.load_dialect(testDialect)


def test_load_dialect_already_registered():
    ctx = MLContext()
    ctx.register_dialect("test", lambda: testDialect)
    with pytest.raises(
        ValueError,
        match="'test' dialect is already registered, use "
        "'load_registered_dialect' instead",
    ):
        ctx.load_dialect(testDialect)


def test_get_optional_op_registered():
    ctx = MLContext()
    ctx.register_dialect("test", lambda: testDialect)
    assert ctx.get_optional_op("test.dummy") == DummyOp
    assert ctx.get_optional_op("test.dummy2") is None
    assert list(ctx.loaded_dialects) == [testDialect]
    assert list(ctx.registered_dialect_names) == ["test"]


def test_get_optional_attr_registered():
    ctx = MLContext()
    ctx.register_dialect("test", lambda: testDialect)
    assert ctx.get_optional_attr("test.dummy_attr") == DummyAttr
    assert ctx.get_optional_attr("test.dummy_attr2") is None
    assert list(ctx.loaded_dialects) == [testDialect]
    assert list(ctx.registered_dialect_names) == ["test"]
