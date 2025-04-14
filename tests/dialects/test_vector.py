import pytest

from xdsl.dialects.builtin import (
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
    CreateMaskOp,
    ExtractElementOp,
    ExtractOp,
    FMAOp,
    InsertElementOp,
    InsertOp,
    LoadOp,
    MaskedLoadOp,
    MaskedStoreOp,
    PrintOp,
    StoreOp,
)
from xdsl.ir import Attribute, OpResult, SSAValue
from xdsl.utils.test_value import create_ssa_value


def get_MemRef_SSAVal(
    referenced_type: Attribute, shape: list[int | IntAttr]
) -> SSAValue:
    memref_type = MemRefType(referenced_type, shape)
    return create_ssa_value(memref_type)


def get_Vector_SSAVal(
    referenced_type: Attribute, shape: list[int | IntAttr]
) -> SSAValue:
    vector_type = VectorType(referenced_type, shape)
    return create_ssa_value(vector_type)


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
    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())
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

    assert store.base is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices == ()


def test_vector_store_i32_with_dimensions():
    vector_ssa_value = get_Vector_SSAVal(i32, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])

    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())
    store = StoreOp.get(vector_ssa_value, memref_ssa_value, [index1, index2])

    assert store.base is memref_ssa_value
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
    index1 = create_ssa_value(IndexType())
    broadcast = BroadcastOp.get(index1)

    assert type(broadcast.results[0]) is OpResult
    assert type(broadcast.results[0].type) is VectorType
    assert broadcast.source is index1


def test_vector_broadcast_verify_type_matching():
    index1 = create_ssa_value(IndexType())
    res_vector_type = VectorType(i64, [1])

    broadcast = BroadcastOp.build(operands=[index1], result_types=[res_vector_type])

    with pytest.raises(
        Exception,
        match="Source operand and result vector must have the same element type.",
    ):
        broadcast.verify()


def test_vector_fma():
    i32_vector_type = VectorType(i32, [1])

    lhs_vector_ssa_value = create_ssa_value(i32_vector_type)
    rhs_vector_ssa_value = create_ssa_value(i32_vector_type)
    acc_vector_ssa_value = create_ssa_value(i32_vector_type)

    fma = FMAOp.get(lhs_vector_ssa_value, rhs_vector_ssa_value, acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].type) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_fma_with_dimensions():
    i32_vector_type = VectorType(i32, [2, 3])

    lhs_vector_ssa_value = create_ssa_value(i32_vector_type)
    rhs_vector_ssa_value = create_ssa_value(i32_vector_type)
    acc_vector_ssa_value = create_ssa_value(i32_vector_type)

    fma = FMAOp.get(lhs_vector_ssa_value, rhs_vector_ssa_value, acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].type) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_masked_load():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    maskedload = MaskedLoadOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, passthrough_vector_ssa_value
    )

    assert type(maskedload.results[0]) is OpResult
    assert type(maskedload.results[0].type) is VectorType
    assert maskedload.indices == ()


def test_vector_masked_load_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())

    maskedload = MaskedLoadOp.get(
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

    maskedload = MaskedLoadOp.build(
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

    maskedload = MaskedLoadOp.build(
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

    maskedload = MaskedLoadOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, passthrough_vector_ssa_value
    )

    with pytest.raises(Exception, match="Expected an index for each memref dimension."):
        maskedload.verify()


def test_vector_masked_store():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    maskedstore = MaskedStoreOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    assert maskedstore.base is memref_ssa_value
    assert maskedstore.mask is mask_vector_ssa_value
    assert maskedstore.value_to_store is value_to_store_vector_ssa_value
    assert maskedstore.indices == ()


def test_vector_masked_store_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())

    maskedstore = MaskedStoreOp.get(
        memref_ssa_value,
        [index1, index2],
        mask_vector_ssa_value,
        value_to_store_vector_ssa_value,
    )

    assert maskedstore.base is memref_ssa_value
    assert maskedstore.mask is mask_vector_ssa_value
    assert maskedstore.value_to_store is value_to_store_vector_ssa_value
    assert maskedstore.indices[0] is index1
    assert maskedstore.indices[1] is index2


def test_vector_masked_store_verify_memref_value_to_store_type_matching():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal(i64, [1])

    maskedstore = MaskedStoreOp.get(
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

    maskedstore = MaskedStoreOp.get(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    with pytest.raises(Exception, match="Expected an index for each memref dimension."):
        maskedstore.verify()


def test_vector_print():
    vector_ssa_value = get_Vector_SSAVal(i32, [1])

    print = PrintOp.get(vector_ssa_value)

    assert print.source is vector_ssa_value


def test_vector_create_mask():
    create_mask = CreateMaskOp.get([])

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].type) is VectorType
    assert create_mask.mask_dim_sizes == ()


def test_vector_create_mask_with_dimensions():
    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())

    create_mask = CreateMaskOp.get([index1, index2])

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].type) is VectorType
    assert create_mask.mask_dim_sizes[0] is index1
    assert create_mask.mask_dim_sizes[1] is index2


def test_vector_create_mask_verify_indexing_exception():
    mask_vector_type = VectorType(i1, [2, 3])

    create_mask = CreateMaskOp.build(operands=[[]], result_types=[mask_vector_type])

    with pytest.raises(
        Exception,
        match="Expected an operand value for each dimension of resultant mask.",
    ):
        create_mask.verify()


def test_vector_extract_element_verify_vector_rank_0_or_1():
    vector_type = VectorType(IndexType(), [3, 3])

    vector = create_ssa_value(vector_type)
    position = create_ssa_value(IndexType())
    extract_element = ExtractElementOp(vector, position)

    with pytest.raises(Exception, match="Unexpected >1 vector rank."):
        extract_element.verify()


def test_vector_extract_element_construction_1d():
    vector_type = VectorType(IndexType(), [3])

    vector = create_ssa_value(vector_type)
    position = create_ssa_value(IndexType())

    extract_element = ExtractElementOp(vector, position)

    assert extract_element.vector is vector
    assert extract_element.position is position
    assert extract_element.result.type == vector_type.element_type


def test_vector_extract_element_1d_verify_non_empty_position():
    vector_type = VectorType(IndexType(), [3])

    vector = create_ssa_value(vector_type)

    extract_element = ExtractElementOp(vector)

    with pytest.raises(Exception, match="Expected position for 1-D vector."):
        extract_element.verify()


def test_vector_extract_element_construction_0d():
    vector_type = VectorType(IndexType(), [])

    vector = create_ssa_value(vector_type)

    extract_element = ExtractElementOp(vector)

    assert extract_element.vector is vector
    assert extract_element.position is None
    assert extract_element.result.type == vector_type.element_type


def test_vector_extract_element_0d_verify_empty_position():
    vector_type = VectorType(IndexType(), [])

    vector = create_ssa_value(vector_type)
    position = create_ssa_value(IndexType())

    extract_element = ExtractElementOp(vector, position)

    with pytest.raises(
        Exception, match="Expected position to be empty with 0-D vector."
    ):
        extract_element.verify()


def test_vector_extract():
    vector_type = VectorType(i32, [1, 2, 3, 4])
    vector = create_ssa_value(vector_type)
    dim1 = create_ssa_value(i32)
    dim2 = create_ssa_value(i32)
    dimensions = [0, dim1, 1, dim2]

    extract = ExtractOp(vector, dimensions, i32)
    assert extract.vector == vector
    assert extract.dynamic_position == (
        dim1,
        dim2,
    )
    assert tuple(extract.static_position.iter_values()) == (
        0,
        extract.DYNAMIC_INDEX,
        1,
        extract.DYNAMIC_INDEX,
    )
    assert extract.result.type == i32


def test_vector_insert_element_verify_vector_rank_0_or_1():
    vector_type = VectorType(IndexType(), [3, 3])

    source = create_ssa_value(IndexType())
    dest = create_ssa_value(vector_type)
    position = create_ssa_value(IndexType())

    insert_element = InsertElementOp(source, dest, position)

    with pytest.raises(Exception, match="Unexpected >1 vector rank."):
        insert_element.verify()


def test_vector_insert_element_construction_1d():
    vector_type = VectorType(IndexType(), [3])

    source = create_ssa_value(IndexType())
    dest = create_ssa_value(vector_type)
    position = create_ssa_value(IndexType())

    insert_element = InsertElementOp(source, dest, position)

    assert insert_element.source is source
    assert insert_element.dest is dest
    assert insert_element.position is position
    assert insert_element.result.type == vector_type


def test_vector_insert_element_1d_verify_non_empty_position():
    vector_type = VectorType(IndexType(), [3])

    source = create_ssa_value(IndexType())
    dest = create_ssa_value(vector_type)

    insert_element = InsertElementOp(source, dest)

    with pytest.raises(
        Exception,
        match="Expected position for 1-D vector.",
    ):
        insert_element.verify()


def test_vector_insert_element_construction_0d():
    vector_type = VectorType(IndexType(), [])

    source = create_ssa_value(IndexType())
    dest = create_ssa_value(vector_type)

    insert_element = InsertElementOp(source, dest)

    assert insert_element.source is source
    assert insert_element.dest is dest
    assert insert_element.position is None
    assert insert_element.result.type == vector_type


def test_vector_insert_element_0d_verify_empty_position():
    vector_type = VectorType(IndexType(), [])

    source = create_ssa_value(IndexType())
    dest = create_ssa_value(vector_type)
    position = create_ssa_value(IndexType())

    insert_element = InsertElementOp(source, dest, position)

    with pytest.raises(
        Exception,
        match="Expected position to be empty with 0-D vector.",
    ):
        insert_element.verify()


def test_vector_insert():
    value = create_ssa_value(VectorType(i32, [5]))
    dest = create_ssa_value(VectorType(i32, [1, 2, 3, 4, 5]))
    dim1 = create_ssa_value(i32)
    dim2 = create_ssa_value(i32)
    dimensions = [0, dim1, 1, dim2]

    insert = InsertOp(value, dest, dimensions)
    assert insert.source == value
    assert insert.dest == dest
    assert insert.dynamic_position == (
        dim1,
        dim2,
    )
    assert tuple(insert.static_position.iter_values()) == (
        0,
        insert.DYNAMIC_INDEX,
        1,
        insert.DYNAMIC_INDEX,
    )
    assert insert.result.type == dest.type
