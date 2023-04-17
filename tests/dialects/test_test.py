from xdsl.dialects.test import TestOp, TestType


def test_op_from_result_types():
    """Test `from_result_types` method of `TestOp`."""
    op = TestOp.from_result_types(TestType("test1"), TestType("test2"))
    op2 = TestOp.create(result_types=(TestType("test1"), TestType("test2")))
    assert op.is_structurally_equivalent(op2)


def test_op_get_values():
    """Test `get_values` method of `TestOp`."""
    op, (res1, res2) = TestOp.get_values(TestType("test1"), TestType("test2"))
    op2 = TestOp.create(result_types=(TestType("test1"), TestType("test2")))
    assert op.is_structurally_equivalent(op2)
    assert res1 is op.results[0]
    assert res2 is op.results[1]
