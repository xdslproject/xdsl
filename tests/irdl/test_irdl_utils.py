import pytest

from xdsl.dialects.irdl import OperationOp
from xdsl.dialects.irdl.utils import class_name_from_op
from xdsl.ir import Block, Region


@pytest.mark.parametrize(
    "op_name, class_name",
    [
        ("my_operation", "MyOperationOp"),
        ("AlreadyCamelCase", "AlreadyCamelCaseOp"),
        ("nested.name", "NestedNameOp"),
    ],
)
def test_class_name_from_op_name(op_name: str, class_name: str):
    op = OperationOp(op_name, Region(Block()))
    assert class_name_from_op(op) == class_name
