import pytest

from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import DenseArrayBase, i64
from xdsl.utils.exceptions import VerifyException


def test_gep_indices_type_validation():
    size = arith.ConstantOp.from_int_and_width(1, 32)
    opaque_ptr = llvm.AllocaOp(size, builtin.i32)
    ptr_type = llvm.LLVMPointerType()

    gep = llvm.GEPOp.from_mixed_indices(
        opaque_ptr,
        indices=[1],
        pointee_type=builtin.i32,
        result_type=ptr_type,
    )

    gep.properties["rawConstantIndices"] = DenseArrayBase.from_list(i64, [1])

    with pytest.raises(VerifyException, match="rawConstantIndices"):
        gep.verify()
