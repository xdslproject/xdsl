import pytest

from xdsl.dialects.builtin import (
    AffineMapAttr,
    IndexType,
    IntAttr,
    MemRefType,
    VectorType,
    i1,
    i32,
    i64,
)
from xdsl.dialects.vector import (
    BroadcastOp,
    CreatemaskOp,
    FMAOp,
    LoadOp,
    MaskedloadOp,
    MaskedstoreOp,
    PrintOp,
    StoreOp,
    TransferReadOp,
    TransferWriteOp,
)
from xdsl.ir import Attribute, OpResult
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue


def get_MemRef_SSAVal(
    referenced_type: Attribute, shape: list[int | IntAttr]
) -> TestSSAValue:
    memref_type = MemRefType(referenced_type, shape)
    return TestSSAValue(memref_type)


def get_Vector_SSAVal(
    referenced_type: Attribute, shape: list[int | IntAttr]
) -> TestSSAValue:
    vector_type = VectorType(referenced_type, shape)
    return TestSSAValue(vector_type)


def test_vectorType():
    vec = VectorType(i32, [1])

    assert vec.get_num_dims() == 1
    assert vec.get_shape() == (1,)
    assert vec.element_type is i32


def test_vectorType_with_dimensions():
    vec = VectorType(i32, [3, 3, 3])

    assert vec.get_num_dims() == 3
    assert vec.get_shape() == (3, 3, 3)
    assert vec.element_type is i32


def test_vector_load_i32():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    load = LoadOp.get(memref_ssa_value, [])

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].type) is VectorType
    assert load.indices == ()


def test_vector_load_i32_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal(i32, [2, 3])
    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())
    load = LoadOp.get(memref_ssa_value, [index1, index2])

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].type) is VectorType
    assert load.indices[0] is index1
    assert load.indices[1] is index2


def test_vector_load_verify_type_matching():
    res_vector_type = VectorType(i64, [1])

    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])

    load = LoadOp.build(operands=[memref_ssa_value, []], result_types=[res_vector_type])

    with pytest.raises(
        Exception, match="MemRef element type should match the Vector element type."
    ):
        load.verify()


def test_vector_load_verify_indexing_exception():
    memref_ssa_value = get_MemRef_SSAVal(i32, [2, 3])

    load = LoadOp.get(memref_ssa_value, [])

    with pytest.raises(Exception, match="Expected an index for each dimension."):
        load.verify()


def test_vector_store_i32():
    vector_ssa_value = get_Vector_SSAVal(i32, [1])
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])

    store = StoreOp.get(vector_ssa_value, memref_ssa_value, [])

    assert store.memref is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices == ()


def test_vector_store_i32_with_dimensions():
    vector_ssa_value = get_Vector_SSAVal(i32, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])

    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())
    store = StoreOp.get(vector_ssa_value, memref_ssa_value, [index1, index2])

    assert store.memref is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices[0] is index1
    assert store.indices[1] is index2


def test_vector_store_verify_type_matching():
    vector_ssa_value = get_Vector_SSAVal(i64, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])

    store = StoreOp.get(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(
        Exception, match="MemRef element type should match the Vector element type."
    ):
        store.verify()


def test_vector_store_verify_indexing_exception():
    vector_ssa_value = get_Vector_SSAVal(i32, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])

    store = StoreOp.get(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(Exception, match="Expected an index for each dimension."):
        store.verify()


def test_vector_broadcast():
    index1 = TestSSAValue(IndexType())
    broadcast = BroadcastOp.get(index1)

    assert type(broadcast.results[0]) is OpResult
    assert type(broadcast.results[0].type) is VectorType
    assert broadcast.source is index1


def test_vector_broadcast_verify_type_matching():
    index1 = TestSSAValue(IndexType())
    res_vector_type = VectorType(i64, [1])

    broadcast = BroadcastOp.build(operands=[index1], result_types=[res_vector_type])

    with pytest.raises(
        Exception,
        match="Source operand and result vector must have the same element type.",
    ):
        broadcast.verify()


def test_vector_fma():
    i32_vector_type = VectorType(i32, [1])

    lhs_vector_ssa_value = TestSSAValue(i32_vector_type)
    rhs_vector_ssa_value = TestSSAValue(i32_vector_type)
    acc_vector_ssa_value = TestSSAValue(i32_vector_type)

    fma = FMAOp.get(lhs_vector_ssa_value, rhs_vector_ssa_value, acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].type) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_fma_with_dimensions():
    i32_vector_type = VectorType(i32, [2, 3])

    lhs_vector_ssa_value = TestSSAValue(i32_vector_type)
    rhs_vector_ssa_value = TestSSAValue(i32_vector_type)
    acc_vector_ssa_value = TestSSAValue(i32_vector_type)

    fma = FMAOp.get(lhs_vector_ssa_value, rhs_vector_ssa_value, acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].type) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_fma_verify_res_lhs_type_matching():
    i64_vector_type = VectorType(i64, [1])

    i32_vector_ssa_value = get_Vector_SSAVal(i32, [1])
    i64_vector_ssa_value = get_Vector_SSAVal(i64, [1])

    fma = FMAOp.build(
        operands=[i32_vector_ssa_value, i64_vector_ssa_value, i64_vector_ssa_value],
        result_types=[i64_vector_type],
    )

    message = (
        "Result vector type must match with all source vectors. Found "
        "different types for result vector and lhs vector."
    )
    with pytest.raises(Exception, match=message):
        fma.verify()


def test_vector_fma_verify_res_rhs_type_matching():
    i64_vector_type = VectorType(i64, [1])

    i32_vector_ssa_value = get_Vector_SSAVal(i32, [1])
    i64_vector_ssa_value = get_Vector_SSAVal(i64, [1])

    fma = FMAOp.build(
        operands=[i64_vector_ssa_value, i32_vector_ssa_value, i64_vector_ssa_value],
        result_types=[i64_vector_type],
    )

    message = (
        "Result vector type must match with all source vectors. "
        "Found different types for result vector and rhs vector."
    )

    with pytest.raises(Exception, match=message):
        fma.verify()


def test_vector_fma_verify_res_acc_type_matching():
    i64_vector_type = VectorType(i64, [1])

    i32_vector_ssa_value = get_Vector_SSAVal(i32, [1])
    i64_vector_ssa_value = get_Vector_SSAVal(i64, [1])

    fma = FMAOp.build(
        operands=[i64_vector_ssa_value, i64_vector_ssa_value, i32_vector_ssa_value],
        result_types=[i64_vector_type],
    )

    message = (
        "Result vector type must match with all source vectors. "
        "Found different types for result vector and acc vector."
    )

    with pytest.raises(Exception, match=message):
        fma.verify()


def test_vector_fma_verify_res_lhs_shape_matching():
    i32_vector_type2 = VectorType(i32, [4, 5])

    vector_ssa_value1 = get_Vector_SSAVal(i32, [2, 3])
    vector_ssa_value2 = get_Vector_SSAVal(i32, [4, 5])

    fma = FMAOp.build(
        operands=[vector_ssa_value1, vector_ssa_value2, vector_ssa_value2],
        result_types=[i32_vector_type2],
    )

    message = (
        "Result vector shape must match with all source vector shapes. "
        "Found different shapes for result vector and lhs vector."
    )
    with pytest.raises(Exception, match=message):
        fma.verify()


def test_vector_fma_verify_res_rhs_shape_matching():
    i32_vector_type2 = VectorType(i32, [4, 5])

    vector_ssa_value1 = get_Vector_SSAVal(i32, [2, 3])
    vector_ssa_value2 = get_Vector_SSAVal(i32, [4, 5])

    fma = FMAOp.build(
        operands=[vector_ssa_value2, vector_ssa_value1, vector_ssa_value2],
        result_types=[i32_vector_type2],
    )

    message = (
        "Result vector shape must match with all source vector shapes. "
        "Found different shapes for result vector and rhs vector."
    )
    with pytest.raises(Exception, match=message):
        fma.verify()


def test_vector_fma_verify_res_acc_shape_matching():
    i32_vector_type2 = VectorType(i32, [4, 5])

    vector_ssa_value1 = get_Vector_SSAVal(i32, [2, 3])
    vector_ssa_value2 = get_Vector_SSAVal(i32, [4, 5])

    fma = FMAOp.build(
        operands=[vector_ssa_value2, vector_ssa_value2, vector_ssa_value1],
        result_types=[i32_vector_type2],
    )

    message = (
        "Result vector shape must match with all source vector shapes. "
        "Found different shapes for result vector and acc vector."
    )
    with pytest.raises(Exception, match=message):
        fma.verify()


def test_vector_masked_load():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    maskedload = MaskedloadOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, passthrough_vector_ssa_value
    )

    assert type(maskedload.results[0]) is OpResult
    assert type(maskedload.results[0].type) is VectorType
    assert maskedload.indices == ()


def test_vector_masked_load_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())

    maskedload = MaskedloadOp.get(
        memref_ssa_value,
        [index1, index2],
        mask_vector_ssa_value,
        passthrough_vector_ssa_value,
    )

    assert type(maskedload.results[0]) is OpResult
    assert type(maskedload.results[0].type) is VectorType
    assert maskedload.indices[0] is index1
    assert maskedload.indices[1] is index2


def test_vector_masked_load_verify_memref_res_type_matching():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    i64_res_vector_type = VectorType(i64, [1])

    maskedload = MaskedloadOp.build(
        operands=[
            memref_ssa_value,
            [],
            mask_vector_ssa_value,
            passthrough_vector_ssa_value,
        ],
        result_types=[i64_res_vector_type],
    )

    message = (
        "MemRef element type should match the result vector and passthrough "
        "vector element type. Found different element types for memref and result."
    )
    with pytest.raises(Exception, match=message):
        maskedload.verify()


def test_vector_masked_load_verify_memref_passthrough_type_matching():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal(i64, [1])

    i64_res_vector_type = VectorType(i32, [1])

    maskedload = MaskedloadOp.build(
        operands=[
            memref_ssa_value,
            [],
            mask_vector_ssa_value,
            passthrough_vector_ssa_value,
        ],
        result_types=[i64_res_vector_type],
    )

    message = (
        "MemRef element type should match the result vector and passthrough "
        "vector element type. Found different element types for memref and passthrough."
    )

    with pytest.raises(Exception, match=message):
        maskedload.verify()


def test_vector_masked_load_verify_indexing_exception():
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [2])
    passthrough_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    maskedload = MaskedloadOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, passthrough_vector_ssa_value
    )

    with pytest.raises(Exception, match="Expected an index for each memref dimension."):
        maskedload.verify()


def test_vector_masked_store():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    maskedstore = MaskedstoreOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    assert maskedstore.memref is memref_ssa_value
    assert maskedstore.mask is mask_vector_ssa_value
    assert maskedstore.value_to_store is value_to_store_vector_ssa_value
    assert maskedstore.indices == ()


def test_vector_masked_store_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())

    maskedstore = MaskedstoreOp.get(
        memref_ssa_value,
        [index1, index2],
        mask_vector_ssa_value,
        value_to_store_vector_ssa_value,
    )

    assert maskedstore.memref is memref_ssa_value
    assert maskedstore.mask is mask_vector_ssa_value
    assert maskedstore.value_to_store is value_to_store_vector_ssa_value
    assert maskedstore.indices[0] is index1
    assert maskedstore.indices[1] is index2


def test_vector_masked_store_verify_memref_value_to_store_type_matching():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal(i64, [1])

    maskedstore = MaskedstoreOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    message = (
        "MemRef element type should match the stored vector type. "
        "Obtained types were i32 and i64."
    )
    with pytest.raises(Exception, match=message):
        maskedstore.verify()


def test_vector_masked_store_verify_indexing_exception():
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [2])
    value_to_store_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    maskedstore = MaskedstoreOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    with pytest.raises(Exception, match="Expected an index for each memref dimension."):
        maskedstore.verify()


def test_vector_print():
    vector_ssa_value = get_Vector_SSAVal(i32, [1])

    print = PrintOp.get(vector_ssa_value)

    assert print.source is vector_ssa_value


def test_vector_create_mask():
    create_mask = CreatemaskOp.get([])

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].type) is VectorType
    assert create_mask.mask_operands == ()


def test_vector_create_mask_with_dimensions():
    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())

    create_mask = CreatemaskOp.get([index1, index2])

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].type) is VectorType
    assert create_mask.mask_operands[0] is index1
    assert create_mask.mask_operands[1] is index2


def test_vector_create_mask_verify_indexing_exception():
    mask_vector_type = VectorType(i1, [2, 3])

    create_mask = CreatemaskOp.build(operands=[[]], result_types=[mask_vector_type])

    with pytest.raises(
        Exception,
        match="Expected an operand value for each dimension of resultant mask.",
    ):
        create_mask.verify()


def test_vector_transfer_write_construction():
    x = AffineExpr.dimension(0)
    vector_type = VectorType(IndexType(), [3])
    memref_type = MemRefType(IndexType(), [3, 3])
    # (x, y) -> x
    permutation_map = AffineMapAttr(AffineMap(2, 0, (x,)))

    vector = TestSSAValue(vector_type)
    source = TestSSAValue(memref_type)
    index = TestSSAValue(IndexType())

    transfer_write = TransferWriteOp(
        vector,
        source,
        [index, index],
        permutation_map=permutation_map,
    )

    transfer_write.verify()

    assert transfer_write.vector is vector
    assert transfer_write.source is source
    assert len(transfer_write.indices) == 2
    assert transfer_write.indices[0] is index
    assert transfer_write.permutation_map is permutation_map


def test_vector_transfer_read_construction():
    x = AffineExpr.dimension(0)
    vector_type = VectorType(IndexType(), [3])
    memref_type = MemRefType(IndexType(), [3, 3])
    permutation_map = AffineMapAttr(AffineMap(2, 0, (x,)))

    source = TestSSAValue(memref_type)
    index = TestSSAValue(IndexType())
    padding = TestSSAValue(IndexType())

    transfer_read = TransferReadOp(
        source,
        [index, index],
        padding,
        vector_type,
        permutation_map=permutation_map,
    )

    transfer_read.verify()

    assert transfer_read.source is source
    assert len(transfer_read.indices) == 2
    assert transfer_read.indices[0] is index
    assert transfer_read.permutation_map is permutation_map
