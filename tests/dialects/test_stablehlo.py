from xdsl.dialects.stablehlo import AddOp
from xdsl.ir import Attribute, OpResult

try:
    from typing import assert_type
    # assert_type is only supported on python 3.11 and above
    # https://docs.python.org/3/library/typing.html#typing.assert_type
except ImportError:
    # if we cannot use typing.assert_type
    # use typing_extensions.assert_type
    # https://typing-extensions.readthedocs.io/en/latest/#typing_extensions.assert_type
    from typing_extensions import assert_type


def test_type_checking_for_elementwise_operation():
    assert_type(
        AddOp.result,
        OpResult[Attribute],
    )
