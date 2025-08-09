from collections.abc import Sequence

import pytest

from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    BoolAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    MemRefType,
    TensorType,
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
    TransferReadOp,
    TransferWriteOp,
    VectorTransferOperation,
)
from xdsl.ir import Attribute, OpResult, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
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
    assert vec.get_num_scalable_dims() == 0
    assert vec.get_scalable_dims() == (False,)


def test_vectorType_with_dimensions():
    vec = VectorType(i32, [3, 3, 3])

    assert vec.get_num_dims() == 3
    assert vec.get_shape() == (3, 3, 3)
    assert vec.element_type is i32
    assert vec.get_num_scalable_dims() == 0
    assert vec.get_scalable_dims() == (
        False,
        False,
        False,
    )


def test_vectorType_with_scalable_dims():
    vec = VectorType(
        i32,
        [3, 3, 3],
        scalable_dims=ArrayAttr(
            (
                BoolAttr.from_bool(False),
                BoolAttr.from_bool(True),
                BoolAttr.from_bool(False),
            )
        ),
    )

    assert vec.get_num_dims() == 3
    assert vec.get_shape() == (3, 3, 3)
    assert vec.element_type is i32
    assert vec.get_num_scalable_dims() == 1
    assert vec.get_scalable_dims() == (
        False,
        True,
        False,
    )


def test_vector_load_i32():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    load = LoadOp(memref_ssa_value, [], VectorType(i32, ()))

    assert type(load.results[0]) is OpResult
    assert type(load.results[0].type) is VectorType
    assert load.indices == ()


def test_vector_load_i32_with_dimensions():
    memref_ssa_value = get_MemRef_SSAVal(i32, [2, 3])
    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())
    load = LoadOp(memref_ssa_value, [index1, index2], VectorType(i32, (3,)))

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

    load = LoadOp(memref_ssa_value, [], VectorType(i32, ()))

    with pytest.raises(Exception, match="Expected an index for each dimension."):
        load.verify()


def test_vector_store_i32():
    vector_ssa_value = get_Vector_SSAVal(i32, [1])
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])

    store = StoreOp(vector_ssa_value, memref_ssa_value, [])

    assert store.base is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices == ()


def test_vector_store_i32_with_dimensions():
    vector_ssa_value = get_Vector_SSAVal(i32, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])

    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())
    store = StoreOp(vector_ssa_value, memref_ssa_value, [index1, index2])

    assert store.base is memref_ssa_value
    assert store.vector is vector_ssa_value
    assert store.indices[0] is index1
    assert store.indices[1] is index2


def test_vector_store_verify_type_matching():
    vector_ssa_value = get_Vector_SSAVal(i64, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])

    store = StoreOp(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(
        Exception, match="MemRef element type should match the Vector element type."
    ):
        store.verify()


def test_vector_store_verify_indexing_exception():
    vector_ssa_value = get_Vector_SSAVal(i32, [2, 3])
    memref_ssa_value = get_MemRef_SSAVal(i32, [4, 5])

    store = StoreOp(vector_ssa_value, memref_ssa_value, [])

    with pytest.raises(Exception, match="Expected an index for each dimension."):
        store.verify()


def test_vector_broadcast():
    index1 = create_ssa_value(IndexType())
    broadcast = BroadcastOp(index1, VectorType(IndexType(), ()))

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

    fma = FMAOp(lhs_vector_ssa_value, rhs_vector_ssa_value, acc_vector_ssa_value)

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

    fma = FMAOp(lhs_vector_ssa_value, rhs_vector_ssa_value, acc_vector_ssa_value)

    assert type(fma.results[0]) is OpResult
    assert type(fma.results[0].type) is VectorType
    assert fma.lhs is lhs_vector_ssa_value
    assert fma.rhs is rhs_vector_ssa_value
    assert fma.acc is acc_vector_ssa_value


def test_vector_masked_load():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    passthrough_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    maskedload = MaskedLoadOp(
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

    maskedload = MaskedLoadOp(
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

    maskedload = MaskedLoadOp(
        memref_ssa_value, [], mask_vector_ssa_value, passthrough_vector_ssa_value
    )

    with pytest.raises(Exception, match="Expected an index for each memref dimension."):
        maskedload.verify()


def test_vector_masked_store():
    memref_ssa_value = get_MemRef_SSAVal(i32, [1])
    mask_vector_ssa_value = get_Vector_SSAVal(i1, [1])
    value_to_store_vector_ssa_value = get_Vector_SSAVal(i32, [1])

    maskedstore = MaskedStoreOp(
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

    maskedstore = MaskedStoreOp(
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

    maskedstore = MaskedStoreOp(
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

    maskedstore = MaskedStoreOp(
        memref_ssa_value, [], mask_vector_ssa_value, value_to_store_vector_ssa_value
    )

    with pytest.raises(Exception, match="Expected an index for each memref dimension."):
        maskedstore.verify()


def test_vector_print():
    vector_ssa_value = get_Vector_SSAVal(i32, [1])

    print = PrintOp(vector_ssa_value)

    assert print.source is vector_ssa_value


def test_vector_create_mask():
    create_mask = CreateMaskOp([], VectorType(i1, []))

    assert type(create_mask.results[0]) is OpResult
    assert type(create_mask.results[0].type) is VectorType
    assert create_mask.mask_dim_sizes == ()


def test_vector_create_mask_with_dimensions():
    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())

    create_mask = CreateMaskOp([index1, index2], VectorType(i1, [2]))

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


@pytest.mark.parametrize(
    "perm_map,input_shape,input_scalable_dims,output_shape,output_scalable_dims",
    [
        (
            # identity no scalable dims
            AffineMap.from_callable(lambda d0, d1: (d0, d1)),
            (2, 3),
            (False, False),
            (2, 3),
            (False, False),
        ),
        (
            # identity with scalable dims
            AffineMap.from_callable(lambda d0, d1: (d0, d1)),
            (2, 3),
            (True, False),
            (2, 3),
            (True, False),
        ),
        (
            # inverse permutation
            AffineMap.from_callable(lambda d0, d1: (d1, d0)),
            (2, 3),
            (True, False),
            (3, 2),
            (False, True),
        ),
        (
            # unused dims
            AffineMap.from_callable(lambda d0, d1, d2: (d1, d0)),
            (2, 3),
            (True, False),
            (3, 2),
            (False, True),
        ),
    ],
)
def test_infer_transfer_op_mask_type(
    perm_map: AffineMap,
    input_shape: Sequence[int],
    input_scalable_dims: Sequence[bool],
    output_shape: Sequence[int],
    output_scalable_dims: Sequence[bool],
):
    vec_type = VectorType(
        i32, input_shape, ArrayAttr(BoolAttr.from_bool(b) for b in input_scalable_dims)
    )
    assert VectorTransferOperation.infer_transfer_op_mask_type(
        vec_type, perm_map
    ) == VectorType(
        i1,
        output_shape,
        ArrayAttr(BoolAttr.from_bool(b) for b in output_scalable_dims),
    )


@pytest.mark.parametrize(
    "shaped_type, vector_type, expected_map",
    [
        (
            # 0-d transfer
            MemRefType(i32, ()),
            VectorType(i32, (1,)),
            AffineMap.constant_map(0),
        ),
        (
            # 1-d transfer to 1-d vector
            MemRefType(i32, (10,)),
            VectorType(i32, (5,)),
            AffineMap.identity(1),
        ),
        (
            # 2-d transfer to 1-d vector (minor identity)
            MemRefType(i32, (10, 20)),
            VectorType(i32, (5,)),
            AffineMap.from_callable(lambda d0, d1: (d1,)),
        ),
        (
            # 3-d transfer to 2-d vector (minor identity)
            MemRefType(i32, (10, 20, 30)),
            VectorType(i32, (5, 6)),
            AffineMap.from_callable(lambda d0, d1, d2: (d1, d2)),
        ),
        (
            # Transfer with vector element type
            MemRefType(VectorType(i32, (4, 3)), (10, 20)),
            VectorType(i32, (5, 6, 4, 3)),
            AffineMap.identity(2),
        ),
        (
            # Tensor type
            TensorType(i32, (10, 20)),
            VectorType(i32, (5,)),
            AffineMap.from_callable(lambda d0, d1: (d1,)),
        ),
    ],
)
def test_get_transfer_minor_identity_map(
    shaped_type: TensorType | MemRefType,
    vector_type: VectorType,
    expected_map: AffineMap,
):
    result_map = VectorTransferOperation.get_transfer_minor_identity_map(
        shaped_type, vector_type
    )
    assert result_map == expected_map


def test_vector_transfer_write_construction():
    x = AffineExpr.dimension(0)
    vector_type = VectorType(IndexType(), [3])
    memref_type = MemRefType(IndexType(), [3, 3])
    # (x, y) -> x
    permutation_map = AffineMapAttr(AffineMap(2, 0, (x,)))
    in_bounds = ArrayAttr([IntegerAttr.from_bool(False)] * vector_type.get_num_dims())

    vector = create_ssa_value(vector_type)
    source = create_ssa_value(memref_type)
    index = create_ssa_value(IndexType())

    transfer_write = TransferWriteOp(
        vector,
        source,
        [index, index],
        in_bounds,
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
    in_bounds = ArrayAttr([IntegerAttr.from_bool(False)] * vector_type.get_num_dims())

    source = create_ssa_value(memref_type)
    index = create_ssa_value(IndexType())
    padding = create_ssa_value(IndexType())

    transfer_read = TransferReadOp(
        source,
        [index, index],
        padding,
        vector_type,
        in_bounds,
        permutation_map=permutation_map,
    )

    transfer_read.verify()

    assert transfer_read.source is source
    assert len(transfer_read.indices) == 2
    assert transfer_read.indices[0] is index
    assert transfer_read.permutation_map is permutation_map
