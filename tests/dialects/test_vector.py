import pytest

from xdsl.dialects.builtin import i32, i64, IntegerType, IndexType, VectorType
from xdsl.dialects.memref import MemRefType
from xdsl.dialects.vector import Load, Store
from xdsl.ir import OpResult


def test_vectorType():
    vec = VectorType.from_type_and_list(i32)

    assert vec.get_num_dims() == 1
    assert vec.get_shape() == [1]
    assert vec.element_type is i32


def test_vectorType_with_dimensions():
    vec = VectorType.from_type_and_list(i32, [3, 3, 3])

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
    i32_memref_type = MemRefType.from_type_and_list(i32)
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    load = Load.get(memref_ssa_value, [])

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].typ) is VectorType
    assert load.indices == []


def test_vector_load_i32_with_dimensions():
    i32_memref_type = MemRefType.from_type_and_list(i32, [2, 3])
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])
    load = Load.get(memref_ssa_value, [index1, index2])

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].typ) is VectorType
    assert load.indices[0] is index1
    assert load.indices[1] is index2


def test_vector_load_verify_type_matching():
    res_vector_type = VectorType.from_type_and_list(i64)

    i32_memref_type = MemRefType.from_type_and_list(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    load = Load.build(operands=[memref_ssa_value, []],
                      result_types=[res_vector_type])

    with pytest.raises(Exception) as exc_info:
        load.verify()
    assert exc_info.value.args[
        0] == "MemRef element type should match the Vector element type."


def test_vector_load_verify_indexing_exception():
    i32_memref_type = MemRefType.from_type_and_list(i32, [2, 3])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    load = Load.get(memref_ssa_value, [])

    with pytest.raises(Exception) as exc_info:
        load.verify()
    assert exc_info.value.args[0] == "Expected an index for each dimension."


def test_vector_store_i32():
    i32_vector_type = VectorType.from_type_and_list(i32)
    vector_ssa_value = OpResult(i32_vector_type, [], [])

    i32_memref_type = MemRefType.from_type_and_list(i32)
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    assert store.memref is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices == []


def test_vector_store_i32_with_dimensions():
    i32_vector_type = VectorType.from_type_and_list(i32, [2, 3])
    vector_ssa_value = OpResult(i32_vector_type, [], [])

    i32_memref_type = MemRefType.from_type_and_list(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])
    store = Store.get(vector_ssa_value, memref_ssa_value, [index1, index2])

    assert store.memref is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices[0] is index1
    assert store.indices[1] is index2


def test_vector_store_verify_type_matching():
    i64_vector_type = VectorType.from_type_and_list(i64, [2, 3])
    vector_ssa_value = OpResult(i64_vector_type, [], [])

    i32_memref_type = MemRefType.from_type_and_list(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(Exception) as exc_info:
        store.verify()
    assert exc_info.value.args[
        0] == "MemRef element type should match the Vector element type."


def test_vector_store_verify_indexing_exception():
    i32_vector_type = VectorType.from_type_and_list(i32, [2, 3])
    vector_ssa_value = OpResult(i32_vector_type, [], [])

    i32_memref_type = MemRefType.from_type_and_list(i32, [4, 5])
    memref_ssa_value = OpResult(i32_memref_type, [], [])

    store = Store.get(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(Exception) as exc_info:
        store.verify()
    assert exc_info.value.args[0] == "Expected an index for each dimension."
