from xdsl.bufferization.op_filter import OpFilter
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition


@irdl_op_definition
class AA(IRDLOperation):
    name = "a.a"


@irdl_op_definition
class AB(IRDLOperation):
    name = "a.b"


@irdl_op_definition
class BA(IRDLOperation):
    name = "b.a"


@irdl_op_definition
class BB(IRDLOperation):
    name = "b.b"


A = Dialect("a", [AA, AB])
B = Dialect("b", [BA, BB])


def test_has_allow():
    op_filter = OpFilter()

    assert not op_filter.has_allow_rule()
    assert op_filter.is_op_allowed(AA)

    op_filter.deny_dialect(A)

    assert not op_filter.has_allow_rule()
    assert not op_filter.is_op_allowed(AA)
    assert op_filter.is_op_allowed(BA)

    op_filter.allow_operation(BA)

    assert op_filter.has_allow_rule()
    assert not op_filter.is_op_allowed(AA)

    assert not op_filter.is_op_allowed(AA)
    assert op_filter.is_op_allowed(BA)
    assert not op_filter.is_op_allowed(BB)


def test_allow_operations_multiple():
    op_filter = OpFilter()
    op_filter.allow_operations({AA, BB})
    assert op_filter.has_allow_rule()
    assert op_filter.is_op_allowed(AA)
    assert not op_filter.is_op_allowed(AB)
    assert not op_filter.is_op_allowed(BA)
    assert op_filter.is_op_allowed(BB)


def test_deny_operations_multiple():
    op_filter = OpFilter()
    op_filter.deny_operations({AA, BB})
    assert not op_filter.has_allow_rule()
    assert not op_filter.is_op_allowed(AA)
    assert op_filter.is_op_allowed(AB)
    assert op_filter.is_op_allowed(BA)
    assert not op_filter.is_op_allowed(BB)


def test_allow_dialect():
    op_filter = OpFilter()
    op_filter.allow_dialect(A)
    assert op_filter.has_allow_rule()
    assert op_filter.is_op_allowed(AA)
    assert op_filter.is_op_allowed(AB)
    assert not op_filter.is_op_allowed(BA)
    assert not op_filter.is_op_allowed(BB)


def test_deny_dialect():
    op_filter = OpFilter()
    op_filter.deny_dialect(A)
    assert not op_filter.has_allow_rule()
    assert not op_filter.is_op_allowed(AA)
    assert not op_filter.is_op_allowed(AB)
    assert op_filter.is_op_allowed(BA)
    assert op_filter.is_op_allowed(BB)


def test_allow_deny_interaction():
    op_filter = OpFilter()
    op_filter.allow_dialect(A)
    op_filter.deny_operation(AA)
    assert op_filter.has_allow_rule()
    assert not op_filter.is_op_allowed(AA)  # Denied explicitly
    assert op_filter.is_op_allowed(AB)  # Allowed through dialect
    assert not op_filter.is_op_allowed(BA)  # Not allowed (no allow rule)
    assert not op_filter.is_op_allowed(BB)  # Not allowed (no allow rule)
