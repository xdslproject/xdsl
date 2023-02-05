import pytest

from xdsl.dialects.builtin import i1, i32, i64, IntegerType, IndexType, VectorType
from xdsl.dialects.memref import MemRefType
from xdsl.dialects.vector import Broadcast, Load, Maskedload, Maskedstore, Store, FMA, Print, Createmask
from xdsl.ir import OpResult


def test_vectorType():
    vec = VectorType.from_element_type_and_shape(i32, [1])

    assert vec.get_num_dims() == 1
    assert vec.get_shape() == [1]
    assert vec.element_type is i32


def test_vectorType_with_dimensions():
    vec = VectorType.from_element_type_and_shape(i32, [3, 3, 3])

    assert vec.get_num_dims() == 3
    assert vec.get_shape() == [3, 3, 3]
    assert vec.element_type is i32


def test_vectorType_from_params():
    my_i32 = IntegerType.from_width(32)
    vec = VectorType.from_params(my_i32)

    assert vec.get_num_dims() == 1
    assert vec.get_shape() == [1]
    assert vec.element_type is my_i32


def test_vector_load_i32():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    load = Load.get(memref_ssa_value, [])

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].typ) is VectorType
    assert load.indices == ()


def test_vector_load_i32_with_dimensions():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])
    load = Load.get(memref_ssa_value, [index1, index2])

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].typ) is VectorType
    assert load.indices[0] is index1
    assert load.indices[1] is index2


def test_vector_load_verify_type_matching():
    res_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    load = Load.build(operands=[memref_ssa_value, []],
                      result_types=[res_vector_type])

    with pytest.raises(Exception) as exc_info:
        load.verify()
    assert exc_info.value.args[
        0] == "MemRef element type should match the Vector element type."


def test_vector_load_verify_indexing_exception():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    load = Load.get(memref_ssa_value, [])

    with pytest.raises(Exception) as exc_info:
        load.verify()
    assert exc_info.value.args[0] == "Expected an index for each dimension."


def test_vector_store_i32():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [1])
    vector_ssa_value = OpResult(i32_vector_type, [], [])

    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    assert store.memref is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices == ()


def test_vector_store_i32_with_dimensions():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [2, 3])
    vector_ssa_value = OpResult(i32_vector_type, [], [])

    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])
    store = Store.get(vector_ssa_value, memref_ssa_value, [index1, index2])

    assert store.memref is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices[0] is index1
    assert store.indices[1] is index2


def test_vector_store_verify_type_matching():
    i64_vector_type = VectorType.from_element_type_and_shape(i64, [2, 3])
    vector_ssa_value = OpResult(i64_vector_type, [], [])

    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(Exception) as exc_info:
        store.verify()
    assert exc_info.value.args[
        0] == "MemRef element type should match the Vector element type."


def test_vector_store_verify_indexing_exception():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [2, 3])
    vector_ssa_value = OpResult(i32_vector_type, [], [])

    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(Exception) as exc_info:
        store.verify()
    assert exc_info.value.args[0] == "Expected an index for each dimension."


def test_vector_broadcast():
    index1 = OpResult(IndexType, [], [])
    broadcast = Broadcast.get(index1)

    assert type(broadcast.results[0]) is OpResult
    assert type(broadcast.results[0].typ) is VectorType
    assert broadcast.source is index1


def test_vector_broadcast_verify_type_matching():
    index1 = OpResult(IndexType, [], [])
    res_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    broadcast = Broadcast.build(operands=[index1],
                                result_types=[res_vector_type])

    with pytest.raises(Exception) as exc_info:
        broadcast.verify()
    assert exc_info.value.args[
        0] == "Source operand and result vector must have the same element type."


def test_vector_fma():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [1])

    lhs_vector_ssa_value = OpResult(i32_vector_type, [], [])
    rhs_vector_ssa_value = OpResult(i32_vector_type, [], [])
    acc_vector_ssa_value = OpResult(i32_vector_type, [], [])

    fma = FMA.get(lhs_vector_ssa_value, rhs_vector_ssa_value,
                  acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].typ) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_fma_with_dimensions():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [2, 3])

    lhs_vector_ssa_value = OpResult(i32_vector_type, [], [])
    rhs_vector_ssa_value = OpResult(i32_vector_type, [], [])
    acc_vector_ssa_value = OpResult(i32_vector_type, [], [])

    fma = FMA.get(lhs_vector_ssa_value, rhs_vector_ssa_value,
                  acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].typ) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_fma_verify_res_lhs_type_matching():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [1])
    i64_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    i32_vector_ssa_value = OpResult(i32_vector_type, [], [])
    i64_vector_ssa_value = OpResult(i64_vector_type, [], [])

    fma = FMA.build(operands=[
        i32_vector_ssa_value, i64_vector_ssa_value, i64_vector_ssa_value
    ],
                    result_types=[i64_vector_type])

    with pytest.raises(Exception) as exc_info:
        fma.verify()
    assert exc_info.value.args[
        0] == "Result vector type must match with all source vectors. Found different types for result vector and lhs vector."


def test_vector_fma_verify_res_rhs_type_matching():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [1])
    i64_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    i32_vector_ssa_value = OpResult(i32_vector_type, [], [])
    i64_vector_ssa_value = OpResult(i64_vector_type, [], [])

    fma = FMA.build(operands=[
        i64_vector_ssa_value, i32_vector_ssa_value, i64_vector_ssa_value
    ],
                    result_types=[i64_vector_type])

    with pytest.raises(Exception) as exc_info:
        fma.verify()
    assert exc_info.value.args[
        0] == "Result vector type must match with all source vectors. Found different types for result vector and rhs vector."


def test_vector_fma_verify_res_acc_type_matching():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [1])
    i64_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    i32_vector_ssa_value = OpResult(i32_vector_type, [], [])
    i64_vector_ssa_value = OpResult(i64_vector_type, [], [])

    fma = FMA.build(operands=[
        i64_vector_ssa_value, i64_vector_ssa_value, i32_vector_ssa_value
    ],
                    result_types=[i64_vector_type])

    with pytest.raises(Exception) as exc_info:
        fma.verify()
    assert exc_info.value.args[
        0] == "Result vector type must match with all source vectors. Found different types for result vector and acc vector."


def test_vector_fma_verify_res_lhs_shape_matching():
    i32_vector_type1 = VectorType.from_element_type_and_shape(i32, [2, 3])
    i32_vector_type2 = VectorType.from_element_type_and_shape(i32, [4, 5])

    vector_ssa_value1 = OpResult(i32_vector_type1, [], [])
    vector_ssa_value2 = OpResult(i32_vector_type2, [], [])

    fma = FMA.build(
        operands=[vector_ssa_value1, vector_ssa_value2, vector_ssa_value2],
        result_types=[i32_vector_type2])

    with pytest.raises(Exception) as exc_info:
        fma.verify()
    assert exc_info.value.args[
        0] == "Result vector shape must match with all source vector shapes. Found different shapes for result vector and lhs vector."


def test_vector_fma_verify_res_rhs_shape_matching():
    i32_vector_type1 = VectorType.from_element_type_and_shape(i32, [2, 3])
    i32_vector_type2 = VectorType.from_element_type_and_shape(i32, [4, 5])

    vector_ssa_value1 = OpResult(i32_vector_type1, [], [])
    vector_ssa_value2 = OpResult(i32_vector_type2, [], [])

    fma = FMA.build(
        operands=[vector_ssa_value2, vector_ssa_value1, vector_ssa_value2],
        result_types=[i32_vector_type2])

    with pytest.raises(Exception) as exc_info:
        fma.verify()
    assert exc_info.value.args[
        0] == "Result vector shape must match with all source vector shapes. Found different shapes for result vector and rhs vector."


def test_vector_fma_verify_res_acc_shape_matching():
    i32_vector_type1 = VectorType.from_element_type_and_shape(i32, [2, 3])
    i32_vector_type2 = VectorType.from_element_type_and_shape(i32, [4, 5])

    vector_ssa_value1 = OpResult(i32_vector_type1, [], [])
    vector_ssa_value2 = OpResult(i32_vector_type2, [], [])

    fma = FMA.build(
        operands=[vector_ssa_value2, vector_ssa_value2, vector_ssa_value1],
        result_types=[i32_vector_type2])

    with pytest.raises(Exception) as exc_info:
        fma.verify()
    assert exc_info.value.args[
        0] == "Result vector shape must match with all source vector shapes. Found different shapes for result vector and acc vector."


def test_vector_masked_load():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_passthrough_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    passthrough_vector_ssa_value = OpResult(i32_passthrough_vector_type, [],
                                            [])

    maskedload = Maskedload.get(memref_ssa_value, [], mask_vector_ssa_value,
                                passthrough_vector_ssa_value)

    assert type(maskedload.results[0]) is OpResult
    assert type(maskedload.results[0].typ) is VectorType
    assert maskedload.indices == ()


def test_vector_masked_load_with_dimensions():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_passthrough_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    passthrough_vector_ssa_value = OpResult(i32_passthrough_vector_type, [],
                                            [])

    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])

    maskedload = Maskedload.get(memref_ssa_value, [index1, index2],
                                mask_vector_ssa_value,
                                passthrough_vector_ssa_value)

    assert type(maskedload.results[0]) is OpResult
    assert type(maskedload.results[0].typ) is VectorType
    assert maskedload.indices[0] is index1
    assert maskedload.indices[1] is index2


def test_vector_masked_load_verify_memref_res_type_matching():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_passthrough_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    passthrough_vector_ssa_value = OpResult(i32_passthrough_vector_type, [],
                                            [])

    i64_res_vector_type = VectorType.from_element_type_and_shape(i64, [1])

    maskedload = Maskedload.build(operands=[
        memref_ssa_value, [], mask_vector_ssa_value,
        passthrough_vector_ssa_value
    ],
                                  result_types=[i64_res_vector_type])

    with pytest.raises(Exception) as exc_info:
        maskedload.verify()
    assert exc_info.value.args[
        0] == "MemRef element type should match the result vector and passthrough vector element type. Found different element types for memref and result."


def test_vector_masked_load_verify_memref_passthrough_type_matching():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_passthrough_vector_type = VectorType.from_element_type_and_shape(
        i64, [1])
    passthrough_vector_ssa_value = OpResult(i32_passthrough_vector_type, [],
                                            [])

    i64_res_vector_type = VectorType.from_element_type_and_shape(i32, [1])

    maskedload = Maskedload.build(operands=[
        memref_ssa_value, [], mask_vector_ssa_value,
        passthrough_vector_ssa_value
    ],
                                  result_types=[i64_res_vector_type])

    with pytest.raises(Exception) as exc_info:
        maskedload.verify()
    assert exc_info.value.args[
        0] == "MemRef element type should match the result vector and passthrough vector element type. Found different element types for memref and passthrough."


def test_vector_masked_load_verify_indexing_exception():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [2])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_passthrough_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    passthrough_vector_ssa_value = OpResult(i32_passthrough_vector_type, [],
                                            [])

    maskedload = Maskedload.get(memref_ssa_value, [], mask_vector_ssa_value,
                                passthrough_vector_ssa_value)

    with pytest.raises(Exception) as exc_info:
        maskedload.verify()
    assert exc_info.value.args[
        0] == "Expected an index for each memref dimension."


def test_vector_masked_load_verify_result_vector_rank():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_passthrough_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    passthrough_vector_ssa_value = OpResult(i32_passthrough_vector_type, [],
                                            [])

    i32_res_vector_type = VectorType.from_element_type_and_shape(i32, [2, 3])

    maskedload = Maskedload.build(operands=[
        memref_ssa_value, [], mask_vector_ssa_value,
        passthrough_vector_ssa_value
    ],
                                  result_types=[i32_res_vector_type])

    with pytest.raises(Exception) as exc_info:
        maskedload.verify()
    assert exc_info.value.args[0] == "Expected a rank 1 result vector."


def test_vector_masked_load_verify_mask_vector_rank():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [2, 3])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_passthrough_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    passthrough_vector_ssa_value = OpResult(i32_passthrough_vector_type, [],
                                            [])

    maskedload = Maskedload.get(memref_ssa_value, [], mask_vector_ssa_value,
                                passthrough_vector_ssa_value)

    with pytest.raises(Exception) as exc_info:
        maskedload.verify()
    assert exc_info.value.args[0] == "Expected a rank 1 mask vector."


def test_vector_masked_load_verify_mask_vector_type():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i32_mask_vector_type = VectorType.from_element_type_and_shape(i32, [2])
    mask_vector_ssa_value = OpResult(i32_mask_vector_type, [], [])

    i32_passthrough_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    passthrough_vector_ssa_value = OpResult(i32_passthrough_vector_type, [],
                                            [])

    maskedload = Maskedload.get(memref_ssa_value, [], mask_vector_ssa_value,
                                passthrough_vector_ssa_value)

    with pytest.raises(Exception) as exc_info:
        maskedload.verify()
    assert exc_info.value.args[0] == "Expected mask element type to be i1."


def test_vector_masked_store():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_value_to_store_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    value_to_store_vector_ssa_value = OpResult(i32_value_to_store_vector_type,
                                               [], [])

    maskedstore = Maskedstore.get(memref_ssa_value, [], mask_vector_ssa_value,
                                  value_to_store_vector_ssa_value)

    assert maskedstore.memref is memref_ssa_value
    assert maskedstore.mask is mask_vector_ssa_value
    assert maskedstore.value_to_store is value_to_store_vector_ssa_value
    assert maskedstore.indices == ()


def test_vector_masked_store_with_dimensions():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_value_to_store_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    value_to_store_vector_ssa_value = OpResult(i32_value_to_store_vector_type,
                                               [], [])

    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])

    maskedstore = Maskedstore.get(memref_ssa_value, [index1, index2],
                                  mask_vector_ssa_value,
                                  value_to_store_vector_ssa_value)

    assert maskedstore.memref is memref_ssa_value
    assert maskedstore.mask is mask_vector_ssa_value
    assert maskedstore.value_to_store is value_to_store_vector_ssa_value
    assert maskedstore.indices[0] is index1
    assert maskedstore.indices[1] is index2


def test_vector_masked_store_verify_memref_value_to_store_type_matching():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i64_value_to_store_vector_type = VectorType.from_element_type_and_shape(
        i64, [1])
    value_to_store_vector_ssa_value = OpResult(i64_value_to_store_vector_type,
                                               [], [])

    maskedstore = Maskedstore.get(memref_ssa_value, [], mask_vector_ssa_value,
                                  value_to_store_vector_ssa_value)

    with pytest.raises(Exception) as exc_info:
        maskedstore.verify()
    assert exc_info.value.args[
        0] == "MemRef element type should match the stored vector type. Obtained types were !i32 and !i64."


def test_vector_masked_store_verify_indexing_exception():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [2])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_value_to_store_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    value_to_store_vector_ssa_value = OpResult(i32_value_to_store_vector_type,
                                               [], [])

    maskedstore = Maskedload.get(memref_ssa_value, [], mask_vector_ssa_value,
                                 value_to_store_vector_ssa_value)

    with pytest.raises(Exception) as exc_info:
        maskedstore.verify()
    assert exc_info.value.args[
        0] == "Expected an index for each memref dimension."


def test_vector_masked_store_verify_value_to_store_vector_rank():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [1])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_value_to_store_vector_type = VectorType.from_element_type_and_shape(
        i32, [2, 3])
    value_to_store_vector_ssa_value = OpResult(i32_value_to_store_vector_type,
                                               [], [])

    maskedstore = Maskedstore.get(memref_ssa_value, [], mask_vector_ssa_value,
                                  value_to_store_vector_ssa_value)

    with pytest.raises(Exception) as exc_info:
        maskedstore.verify()
    assert exc_info.value.args[0] == "Expected a rank 1 vector to be stored."


def test_vector_masked_store_verify_mask_vector_rank():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i1_mask_vector_type = VectorType.from_element_type_and_shape(i1, [2, 3])
    mask_vector_ssa_value = OpResult(i1_mask_vector_type, [], [])

    i32_value_to_store_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    value_to_store_vector_ssa_value = OpResult(i32_value_to_store_vector_type,
                                               [], [])

    maskedstore = Maskedstore.get(memref_ssa_value, [], mask_vector_ssa_value,
                                  value_to_store_vector_ssa_value)

    with pytest.raises(Exception) as exc_info:
        maskedstore.verify()
    assert exc_info.value.args[0] == "Expected a rank 1 mask vector."


def test_vector_masked_store_verify_mask_vector_type():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    i32_mask_vector_type = VectorType.from_element_type_and_shape(i32, [2])
    mask_vector_ssa_value = OpResult(i32_mask_vector_type, [], [])

    i32_value_to_store_vector_type = VectorType.from_element_type_and_shape(
        i32, [1])
    value_to_store_vector_ssa_value = OpResult(i32_value_to_store_vector_type,
                                               [], [])

    maskedstore = Maskedstore.get(memref_ssa_value, [], mask_vector_ssa_value,
                                  value_to_store_vector_ssa_value)

    with pytest.raises(Exception) as exc_info:
        maskedstore.verify()
    assert exc_info.value.args[0] == "Expected mask element type to be i1."


def test_vector_print():
    i32_vector_type = VectorType.from_element_type_and_shape(i32, [1])
    vector_ssa_value = OpResult(i32_vector_type, [], [])

    print = Print.get(vector_ssa_value)

    assert print.source is vector_ssa_value


def test_vector_create_mask():
    create_mask = Createmask.get([])

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].typ) is VectorType
    assert create_mask.mask_operands == ()


def test_vector_create_mask_with_dimensions():
    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])

    create_mask = Createmask.get([index1, index2])

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].typ) is VectorType
    assert create_mask.mask_operands[0] is index1
    assert create_mask.mask_operands[1] is index2


def test_vector_create_mask_verify_mask_vector_type():
    mask_vector_type = VectorType.from_element_type_and_shape(i32, [1])

    create_mask = Createmask.build(operands=[[]],
                                   result_types=[mask_vector_type])

    with pytest.raises(Exception) as exc_info:
        create_mask.verify()
    assert exc_info.value.args[0] == "Expected mask element type to be i1."


def test_vector_create_mask_verify_indexing_exception():
    mask_vector_type = VectorType.from_element_type_and_shape(i1, [2, 3])

    create_mask = Createmask.build(operands=[[]],
                                   result_types=[mask_vector_type])

    with pytest.raises(Exception) as exc_info:
        create_mask.verify()
    assert exc_info.value.args[
        0] == "Expected an operand value for each dimension of resultant mask."
