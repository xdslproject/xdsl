from xdsl.ir import OpResult
from xdsl.dialects.builtin import i32, IntegerType, IndexType
from xdsl.dialects.arith import Constant
from xdsl.dialects.memref import (Alloc, Alloca, Dealloc, Dealloca, MemRefType,
                                  Load, Store, Dim, Rank)


def test_memreftype():
    mem1 = MemRefType.from_element_type_and_shape(i32, [1])

    assert mem1.get_num_dims() == 1
    assert mem1.get_shape() == [1]
    assert mem1.element_type is i32

    mem2 = MemRefType.from_element_type_and_shape(i32, [3, 3, 3])

    assert mem2.get_num_dims() == 3
    assert mem2.get_shape() == [3, 3, 3]
    assert mem2.element_type is i32

    my_i32 = IntegerType.from_width(32)
    mem3 = MemRefType.from_params(my_i32)

    assert mem3.get_num_dims() == 1
    assert mem3.get_shape() == [1]
    assert mem3.element_type is my_i32


def test_memref_load_i32():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    load = Load.get(memref_ssa_value, [])

    assert load.memref is memref_ssa_value
    assert load.indices == ()
    assert load.res.typ is i32


def test_memref_load_i32_with_dimensions():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])
    load = Load.get(memref_ssa_value, [index1, index2])

    assert load.memref is memref_ssa_value
    assert load.indices[0] is index1
    assert load.indices[1] is index2
    assert load.res.typ is i32


def test_memref_store_i32():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    i32_ssa_value = OpResult(i32, [], [])
    store = Store.get(i32_ssa_value, memref_ssa_value, [])

    assert store.memref is memref_ssa_value
    assert store.indices == ()
    assert store.value is i32_ssa_value


def test_memref_store_i32_with_dimensions():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = OpResult(i32_memref_type, [], [])
    i32_ssa_value = OpResult(i32, [], [])
    index1 = OpResult(IndexType, [], [])
    index2 = OpResult(IndexType, [], [])
    store = Store.get(i32_ssa_value, memref_ssa_value, [index1, index2])

    assert store.memref is memref_ssa_value
    assert store.indices[0] is index1
    assert store.indices[1] is index2
    assert store.value is i32_ssa_value


def test_memref_alloc():
    my_i32 = IntegerType.from_width(32)
    alloc0 = Alloc.get(my_i32, 64, [3, 1, 2])
    alloc1 = Alloc.get(my_i32, 64)

    assert alloc0.dynamic_sizes == ()
    assert type(alloc0.results[0]) is OpResult
    assert alloc0.results[0].typ.get_shape() == [3, 1, 2]
    assert type(alloc0.results[0].typ) is MemRefType
    assert type(alloc1.results[0]) is OpResult
    assert alloc1.results[0].typ.get_shape() == [1]
    assert type(alloc1.results[0].typ) is MemRefType


def test_memref_alloca():
    my_i32 = IntegerType.from_width(32)
    alloc0 = Alloca.get(my_i32, 64, [3, 1, 2])
    alloc1 = Alloca.get(my_i32, 64)

    assert type(alloc0.results[0]) is OpResult
    assert alloc0.results[0].typ.get_shape() == [3, 1, 2]
    assert type(alloc0.results[0].typ) is MemRefType
    assert type(alloc1.results[0]) is OpResult
    assert alloc1.results[0].typ.get_shape() == [1]
    assert type(alloc1.results[0].typ) is MemRefType


def test_memref_dealloc():
    my_i32 = IntegerType.from_width(32)
    alloc0 = Alloc.get(my_i32, 64, [3, 1, 2])
    dealloc0 = Dealloc.get(alloc0)

    assert type(dealloc0.memref) is OpResult


def test_memref_dealloca():
    my_i32 = IntegerType.from_width(32)
    alloc0 = Alloca.get(my_i32, 64, [3, 1, 2])
    dealloc0 = Dealloca.get(alloc0)

    assert type(dealloc0.memref) is OpResult


def test_memref_dim():
    idx = Constant.from_int_and_width(1, IndexType())
    alloc0 = Alloc.get(i32, 64, [3, 1, 2])
    dim_1 = Dim.from_source_and_index(alloc0, idx)

    assert dim_1.source is alloc0.memref
    assert dim_1.index is idx.result
    assert isinstance(dim_1.result.typ, IndexType)


def test_memref_rank():
    alloc0 = Alloc.get(i32, 64, [3, 1, 2])
    dim_1 = Rank.from_memref(alloc0)

    assert dim_1.source is alloc0.memref
    assert isinstance(dim_1.result.typ, IndexType)
