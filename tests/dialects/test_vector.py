from typing import List

import pytest

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    VectorType,
    i1,
    i32,
    i64,
)
from xdsl.dialects.memref import MemRefType
from xdsl.dialects.vector import (
    FMA,
    Broadcast,
    Createmask,
    Load,
    Maskedload,
    Maskedstore,
    Print,
    Store,
)
from xdsl.ir import OpResult
from xdsl.ir.core import Attribute
from xdsl.utils.test_value import TestSSAValue


def get_MemRef_SSAVal_from_element_type_and_shape(
    referenced_type: Attribute, shape: List[int | AnyIntegerAttr]
) -> TestSSAValue:
    memref_type = MemRefType.from_element_type_and_shape(referenced_type, shape)
    return TestSSAValue(memref_type)


def get_Vector_SSAVal_from_element_type_and_shape(
    referenced_type: Attribute, shape: List[int | IntegerAttr[IndexType]]
) -> TestSSAValue:
    vector_type = VectorType.from_element_type_and_shape(referenced_type, shape)
    return TestSSAValue(vector_type)


def test_vectorType():
    vec = VectorType.from_element_type_and_shape(i32, [1])

    assert vec.get_num_dims() == 1
    assert vec.get_shape() == (1,)
    assert vec.element_type is i32


def test_vectorType_with_dimensions():
    vec = VectorType.from_element_type_and_shape(i32, [3, 3, 3])

    assert vec.get_num_dims() == 3
    assert vec.get_shape() == (3, 3, 3)
    assert vec.element_type is i32


def test_vectorType_from_params():
    my_i32 = IntegerType(32)
    vec = VectorType.from_params(my_i32)

    assert vec.get_num_dims() == 1
    assert vec.get_shape() == (1,)
    assert vec.element_type is my_i32


def test_vector_load_i32():
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [1])
    load = Load.get(memref_ssa_value, [])

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].type) is VectorType
    assert load.indices == ()


def test_vector_load_i32_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [2, 3])
    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())
    load = Load.get(memref_ssa_value, [index1, index2])

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].type) is VectorType
    assert load.indices[0] is index1
    assert load.indices[1] is index2


def test_vector_load_verify_type_matching():
    res_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [4, 5])

    load = Load.build(operands=[memref_ssa_value, []], result_types=[res_vector_type])

    with pytest.raises(
        Exception, match="MemRef element type should match the Vector element type."
    ):
        load.verify()


def test_vector_load_verify_indexing_exception():
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [2, 3])

    load = Load.get(memref_ssa_value, [])

    with pytest.raises(Exception, match="Expected an index for each dimension."):
        load.verify()


def test_vector_store_i32():
    vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i32, [1])
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [1])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    assert store.memref is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices == ()


def test_vector_store_i32_with_dimensions():
    vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [4, 5])

    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())
    store = Store.get(vector_ssa_value, memref_ssa_value, [index1, index2])

    assert store.memref is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices[0] is index1
    assert store.indices[1] is index2


def test_vector_store_verify_type_matching():
    vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i64, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [4, 5])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(
        Exception, match="MemRef element type should match the Vector element type."
    ):
        store.verify()


def test_vector_store_verify_indexing_exception():
    vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [4, 5])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(Exception, match="Expected an index for each dimension."):
        store.verify()


def test_vector_broadcast():
    index1 = TestSSAValue(IndexType())
    broadcast = Broadcast.get(index1)

    assert type(broadcast.results[0]) is OpResult
    assert type(broadcast.results[0].type) is VectorType
    assert broadcast.source is index1


def test_vector_broadcast_verify_type_matching():
    index1 = TestSSAValue(IndexType())
    res_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    broadcast = Broadcast.build(operands=[index1], result_types=[res_vector_type])

    with pytest.raises(
        Exception,
        match="Source operand and result vector must have the same element type.",
    ):
        broadcast.verify()


def test_vector_fma():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [1])

    lhs_vector_ssa_value = TestSSAValue(i32_vector_type)
    rhs_vector_ssa_value = TestSSAValue(i32_vector_type)
    acc_vector_ssa_value = TestSSAValue(i32_vector_type)

    fma = FMA.get(lhs_vector_ssa_value, rhs_vector_ssa_value, acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].type) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_fma_with_dimensions():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [2, 3])

    lhs_vector_ssa_value = TestSSAValue(i32_vector_type)
    rhs_vector_ssa_value = TestSSAValue(i32_vector_type)
    acc_vector_ssa_value = TestSSAValue(i32_vector_type)

    fma = FMA.get(lhs_vector_ssa_value, rhs_vector_ssa_value, acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].type) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_fma_verify_res_lhs_type_matching():
    i64_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    i32_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i32, [1])
    i64_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i64, [1])

    fma = FMA.build(
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
    i64_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    i32_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i32, [1])
    i64_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i64, [1])

    fma = FMA.build(
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
    i64_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    i32_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i32, [1])
    i64_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i64, [1])

    fma = FMA.build(
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
    i32_vector_type2 = VectorType.from_element_type_and_shape(i32, [4, 5])

    vector_ssa_value1 = get_Vector_SSAVal_from_element_type_and_shape(i32, [2, 3])
    vector_ssa_value2 = get_Vector_SSAVal_from_element_type_and_shape(i32, [4, 5])

    fma = FMA.build(
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
    i32_vector_type2 = VectorType.from_element_type_and_shape(i32, [4, 5])

    vector_ssa_value1 = get_Vector_SSAVal_from_element_type_and_shape(i32, [2, 3])
    vector_ssa_value2 = get_Vector_SSAVal_from_element_type_and_shape(i32, [4, 5])

    fma = FMA.build(
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
    i32_vector_type2 = VectorType.from_element_type_and_shape(i32, [4, 5])

    vector_ssa_value1 = get_Vector_SSAVal_from_element_type_and_shape(i32, [2, 3])
    vector_ssa_value2 = get_Vector_SSAVal_from_element_type_and_shape(i32, [4, 5])

    fma = FMA.build(
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
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i32, [1]
    )

    maskedload = Maskedload.get(
        memref_ssa_value, [], mask_vector_ssa_value, passthrough_vector_ssa_value
    )

    assert type(maskedload.results[0]) is OpResult
    assert type(maskedload.results[0].type) is VectorType
    assert maskedload.indices == ()


def test_vector_masked_load_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i32, [1]
    )

    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())

    maskedload = Maskedload.get(
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
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i32, [1]
    )

    i64_res_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    maskedload = Maskedload.build(
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
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i64, [1]
    )

    i64_res_vector_type = VectorType.from_element_type_and_shape(i32, [1])

    maskedload = Maskedload.build(
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
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [2])
    passthrough_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i32, [1]
    )

    maskedload = Maskedload.get(
        memref_ssa_value, [], mask_vector_ssa_value, passthrough_vector_ssa_value
    )

    with pytest.raises(Exception, match="Expected an index for each memref dimension."):
        maskedload.verify()


def test_vector_masked_store():
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i32, [1]
    )

    maskedstore = Maskedstore.get(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    assert maskedstore.memref is memref_ssa_value
    assert maskedstore.mask is mask_vector_ssa_value
    assert maskedstore.value_to_store is value_to_store_vector_ssa_value
    assert maskedstore.indices == ()


def test_vector_masked_store_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i32, [1]
    )

    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())

    maskedstore = Maskedstore.get(
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
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i64, [1]
    )

    maskedstore = Maskedstore.get(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    message = (
        "MemRef element type should match the stored vector type. "
        "Obtained types were i32 and i64."
    )
    with pytest.raises(Exception, match=message):
        maskedstore.verify()


def test_vector_masked_store_verify_indexing_exception():
    memref_ssa_value = get_MemRef_SSAVal_from_element_type_and_shape(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i1, [2])
    value_to_store_vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(
        i32, [1]
    )

    maskedstore = Maskedstore.get(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    with pytest.raises(Exception, match="Expected an index for each memref dimension."):
        maskedstore.verify()


def test_vector_print():
    vector_ssa_value = get_Vector_SSAVal_from_element_type_and_shape(i32, [1])

    print = Print.get(vector_ssa_value)

    assert print.source is vector_ssa_value


def test_vector_create_mask():
    create_mask = Createmask.get([])

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].type) is VectorType
    assert create_mask.mask_operands == ()


def test_vector_create_mask_with_dimensions():
    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())

    create_mask = Createmask.get([index1, index2])

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].type) is VectorType
    assert create_mask.mask_operands[0] is index1
    assert create_mask.mask_operands[1] is index2


def test_vector_create_mask_verify_indexing_exception():
    mask_vector_type = VectorType.from_element_type_and_shape(i1, [2, 3])

    create_mask = Createmask.build(operands=[[]], result_types=[mask_vector_type])

    with pytest.raises(
        Exception,
        match="Expected an operand value for each dimension of resultant mask.",
    ):
        create_mask.verify()
