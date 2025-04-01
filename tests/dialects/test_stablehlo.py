from typing_extensions import assert_type

from xdsl.dialects.builtin import (
    AnyTensorType,
    DenseArrayBase,
    IntegerType,
    TensorType,
    i32,
)
from xdsl.dialects.stablehlo import AddOp, TransposeOp
from xdsl.ir import Attribute, OpResult
from xdsl.utils.test_value import TestSSAValue


def test_type_checking_for_elementwise_operation():
    a = TestSSAValue(TensorType(IntegerType(32), []))
    addOp = AddOp(a, a)
    transposeOp = TransposeOp(
        a, DenseArrayBase.from_list(i32, [2, 2]), TensorType(IntegerType(32), [])
    )
    assert_type(
        addOp.result,
        OpResult[Attribute],
    )
    assert_type(
        transposeOp.result,
        OpResult[AnyTensorType],
    )
