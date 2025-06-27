import pytest

from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerType,
    MemRefType,
    NoneAttr,
    StridedLayoutAttr,
    UnrankedMemRefType,
    i32,
    i64,
)
from xdsl.dialects.memref import (
    AllocaOp,
    AllocOp,
    CastOp,
    CopyOp,
    DeallocOp,
    DmaStartOp,
    DmaWaitOp,
    ExtractAlignedPointerAsIndexOp,
    ExtractStridedMetaDataOp,
    LoadOp,
    MemorySpaceCastOp,
    StoreOp,
    SubviewOp,
)
from xdsl.ir import BlockArgument, OpResult
from xdsl.ir.affine import AffineMap
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.test_value import create_ssa_value


def test_memreftype():
    mem1 = MemRefType(i32, [1])

    assert mem1.get_num_dims() == 1
    assert mem1.get_shape() == (1,)
    assert mem1.element_type is i32

    mem2 = MemRefType(i32, [3, 3, 3])

    assert mem2.get_num_dims() == 3
    assert mem2.get_shape() == (3, 3, 3)
    assert mem2.element_type is i32


def test_memref_load_i32():
    i32_memref_type = MemRefType(i32, [1])
    memref_ssa_value = create_ssa_value(i32_memref_type)
    load = LoadOp.get(memref_ssa_value, [])

    assert load.memref is memref_ssa_value
    assert load.indices == ()
    assert load.res.type is i32


def test_memref_load_i32_with_dimensions():
    i32_memref_type = MemRefType(i32, [2, 3])
    memref_ssa_value = create_ssa_value(i32_memref_type)
    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())
    load = LoadOp.get(memref_ssa_value, [index1, index2])

    assert load.memref is memref_ssa_value
    assert load.indices[0] is index1
    assert load.indices[1] is index2
    assert load.res.type is i32


def test_memref_store_i32():
    i32_memref_type = MemRefType(i32, [1])
    memref_ssa_value = create_ssa_value(i32_memref_type)
    i32_ssa_value = create_ssa_value(i32)
    store = StoreOp.get(i32_ssa_value, memref_ssa_value, [])

    assert store.memref is memref_ssa_value
    assert store.indices == ()
    assert store.value is i32_ssa_value


def test_memref_store_i32_with_dimensions():
    i32_memref_type = MemRefType(i32, [2, 3])
    memref_ssa_value = create_ssa_value(i32_memref_type)
    i32_ssa_value = create_ssa_value(i32)
    index1 = create_ssa_value(IndexType())
    index2 = create_ssa_value(IndexType())
    store = StoreOp.get(i32_ssa_value, memref_ssa_value, [index1, index2])

    assert store.memref is memref_ssa_value
    assert store.indices[0] is index1
    assert store.indices[1] is index2
    assert store.value is i32_ssa_value


def test_memref_alloc():
    my_i32 = IntegerType(32)
    my_layout = StridedLayoutAttr(strides=(2, 4, 6), offset=8)
    my_memspace = builtin.IntegerAttr(0, i32)
    alloc0 = AllocOp.get(my_i32, 64, [3, 1, 2])
    alloc1 = AllocOp.get(my_i32, 64)
    alloc2 = AllocOp.get(
        my_i32, 64, [3, 1, 2], layout=my_layout, memory_space=my_memspace
    )

    assert alloc0.dynamic_sizes == ()
    assert type(alloc0.results[0]) is OpResult
    assert type(alloc0.results[0].type) is MemRefType
    assert alloc0.results[0].type.get_shape() == (3, 1, 2)
    assert type(alloc1.results[0]) is OpResult
    assert type(alloc1.results[0].type) is MemRefType
    assert alloc1.results[0].type.get_shape() == (1,)
    assert type(alloc2.results[0]) is OpResult
    assert type(alloc2.results[0].type) is MemRefType
    assert alloc2.results[0].type.get_shape() == (3, 1, 2)
    assert alloc2.results[0].type.layout == my_layout
    assert alloc2.results[0].type.memory_space == my_memspace

    dynamic_sizes_1 = [create_ssa_value(builtin.IndexType()) for _ in range(1)]
    dynamic_sizes_3 = [create_ssa_value(builtin.IndexType()) for _ in range(3)]

    alloc4 = AllocOp.get(my_i32, 64, [3, 1, -1], dynamic_sizes=dynamic_sizes_1)
    alloc5 = AllocOp.get(my_i32, 64, [-1, -1, -1], dynamic_sizes=dynamic_sizes_3)
    alloc6 = AllocOp.get(my_i32, 64, [-1, -1, -1], dynamic_sizes=dynamic_sizes_1)

    assert list(alloc4.operands) == dynamic_sizes_1
    assert list(alloc5.operands) == dynamic_sizes_3
    assert list(alloc6.operands) == dynamic_sizes_1

    alloc4.verify()
    alloc5.verify()

    with pytest.raises(
        VerifyException,
        match="op dimension operand count does not equal memref dynamic dimension count",
    ):
        alloc6.verify()


def test_memref_alloca():
    my_i32 = IntegerType(32)
    my_layout = StridedLayoutAttr(strides=(2, 4, 6), offset=8)
    my_memspace = builtin.IntegerAttr(0, i32)
    alloc0 = AllocaOp.get(my_i32, 64, [3, 1, 2])
    alloc1 = AllocaOp.get(my_i32, 64)
    alloc2 = AllocaOp.get(
        my_i32, 64, [3, 1, 2], layout=my_layout, memory_space=my_memspace
    )

    assert type(alloc0.results[0]) is OpResult
    assert type(alloc0.results[0].type) is MemRefType
    assert alloc0.results[0].type.get_shape() == (3, 1, 2)
    assert type(alloc1.results[0]) is OpResult
    assert type(alloc1.results[0].type) is MemRefType
    assert alloc1.results[0].type.get_shape() == (1,)
    assert type(alloc2.results[0]) is OpResult
    assert type(alloc2.results[0].type) is MemRefType
    assert alloc2.results[0].type.get_shape() == (3, 1, 2)
    assert alloc2.results[0].type.layout == my_layout
    assert alloc2.results[0].type.memory_space == my_memspace

    dynamic_sizes_1 = [create_ssa_value(builtin.IndexType()) for _ in range(1)]
    dynamic_sizes_3 = [create_ssa_value(builtin.IndexType()) for _ in range(3)]

    alloc4 = AllocaOp.get(my_i32, 64, [3, 1, -1], dynamic_sizes=dynamic_sizes_1)
    alloc5 = AllocaOp.get(my_i32, 64, [-1, -1, -1], dynamic_sizes=dynamic_sizes_3)
    alloc6 = AllocaOp.get(my_i32, 64, [-1, -1, -1], dynamic_sizes=dynamic_sizes_1)

    assert list(alloc4.operands) == dynamic_sizes_1
    assert list(alloc5.operands) == dynamic_sizes_3
    assert list(alloc6.operands) == dynamic_sizes_1

    alloc4.verify()
    alloc5.verify()

    with pytest.raises(
        VerifyException,
        match="op dimension operand count does not equal memref dynamic dimension count",
    ):
        alloc6.verify()


def test_memref_dealloc():
    my_i32 = IntegerType(32)
    alloc0 = AllocOp.get(my_i32, 64, [3, 1, 2])
    dealloc0 = DeallocOp.get(alloc0)

    assert type(dealloc0.memref) is OpResult


def test_memref_dim():
    idx = arith.ConstantOp.from_int_and_width(1, IndexType())
    alloc0 = AllocOp.get(i32, 64, [3, 1, 2])
    dim_1 = memref.DimOp.from_source_and_index(alloc0, idx)

    assert dim_1.source is alloc0.memref
    assert dim_1.index is idx.result
    assert isinstance(dim_1.result.type, IndexType)


def test_memref_rank():
    alloc0 = AllocOp.get(i32, 64, [3, 1, 2])
    dim_1 = memref.RankOp.from_memref(alloc0)

    assert dim_1.source is alloc0.memref
    assert isinstance(dim_1.rank.type, IndexType)


def test_memref_ExtractAlignedPointerAsIndexOp():
    ref = AllocOp.get(i32, 64, [64, 64, 64])
    ptr = ExtractAlignedPointerAsIndexOp.get(ref)

    assert ptr.aligned_pointer.type == IndexType()
    assert ptr.source == ref.memref


def test_memref_matmul_verify():
    memref_f64_rank2 = memref.MemRefType(builtin.f64, [-1, -1])

    @builtin.ModuleOp
    @Builder.implicit_region
    def module():
        @Builder.implicit_region((memref_f64_rank2, memref_f64_rank2))
        def matmul(args: tuple[BlockArgument, ...]) -> None:
            a, b = args

            lit0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
            lit1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
            dim_a0 = memref.DimOp.from_source_and_index(a, lit0)
            dim_a1 = memref.DimOp.from_source_and_index(a, lit1)
            dim_b0 = memref.DimOp.from_source_and_index(b, lit0)
            dim_b1 = memref.DimOp.from_source_and_index(b, lit1)
            out = memref.AllocaOp.get(builtin.f64, 0, [-1, -1], [dim_a0, dim_b1])
            # TODO: assert dim_a0 == dim_b1
            lit0_f = arith.ConstantOp(FloatAttr(0.0, builtin.f64))

            @Builder.implicit_region((builtin.IndexType(),))
            def outer_loop(args: tuple[BlockArgument, ...]):
                (i,) = args

                # outer loop start, loop_var = i

                @Builder.implicit_region((builtin.IndexType(),))
                def mid_loop(args: tuple[BlockArgument, ...]):
                    (j,) = args
                    # mid loop start, loop_var = j
                    memref.StoreOp.get(lit0_f, out, [i, j])

                    @Builder.implicit_region((builtin.IndexType(),))
                    def inner_loop(args: tuple[BlockArgument, ...]):
                        (k,) = args
                        # inner loop, loop_var = k
                        elem_a_i_k = memref.LoadOp.get(a, [i, k])
                        elem_b_k_j = memref.LoadOp.get(b, [k, j])
                        mul = arith.MulfOp(elem_a_i_k, elem_b_k_j)
                        out_i_j = memref.LoadOp.get(out, [i, j])
                        new_out_val = arith.AddfOp(out_i_j, mul)
                        memref.StoreOp.get(new_out_val, out, [i, j])
                        scf.YieldOp()

                    scf.ForOp(lit0, dim_a1, lit1, [], inner_loop)
                    scf.YieldOp()

                scf.ForOp(lit0, dim_b0, lit1, [], mid_loop)
                scf.YieldOp()

            scf.ForOp(lit0, dim_a0, lit1, [], outer_loop)

            func.ReturnOp(out)

        func.FuncOp(
            "matmul",
            ((memref_f64_rank2, memref_f64_rank2), (memref_f64_rank2,)),
            matmul,
        )

    # check that it verifies correctly
    module.verify()


def test_memref_extract_strided_metadata():
    # input type
    el_type = i32
    alignment = 8
    shape = [10, 10, 10]
    strides = [1, 2, 2]
    offset = None  # Dynamic offset
    layout = StridedLayoutAttr(strides, offset)
    memory_space = IntAttr(1)
    alloc = AllocOp.get(
        el_type, alignment, shape, layout=layout, memory_space=memory_space
    )
    assert isa(alloc.memref.type, MemRefType)

    extract_op = ExtractStridedMetaDataOp(alloc)
    # output type is expected to have no layout information, and no shape
    expected_type = MemRefType(el_type, [], NoneAttr(), memory_space=memory_space)
    assert extract_op.results[0].type == expected_type


def test_memref_subview_constant_parameters():
    alloc = AllocOp.get(i32, 8, [10, 10, 10])
    assert isa(alloc.memref.type, MemRefType)

    subview = SubviewOp.from_static_parameters(
        alloc, alloc.memref.type, [2, 2, 2], [2, 2, 2], [3, 3, 3]
    )

    assert isinstance(subview, SubviewOp)
    assert isinstance(subview.result.type, MemRefType)
    assert isinstance(subview.result.type.layout, StridedLayoutAttr)
    assert isa(subview.result.type.layout.strides, ArrayAttr[IntAttr])
    out_strides = [i.data for i in subview.result.type.layout.strides.data]
    assert out_strides == [300, 30, 3]
    assert isinstance(subview.result.type.layout.offset, IntAttr)
    assert subview.result.type.layout.offset.data == 222


def test_memref_cast():
    i32_memref_type = MemRefType(i32, [10, 2])
    memref_ssa_value = create_ssa_value(i32_memref_type)

    res_type = UnrankedMemRefType.from_type(i32)

    cast = CastOp.get(memref_ssa_value, res_type)

    assert cast.source is memref_ssa_value
    assert cast.dest.type is res_type


def test_memref_memory_space_cast():
    i32_memref_type = MemRefType(i32, [10, 2], memory_space=builtin.IntegerAttr(1, i32))
    memref_ssa_value = create_ssa_value(i32_memref_type)

    res_type = MemRefType(i32, [10, 2], memory_space=builtin.IntegerAttr(2, i32))
    res_type_wrong_type = MemRefType(
        i64, [10, 2], memory_space=builtin.IntegerAttr(2, i32)
    )
    res_type_wrong_shape = MemRefType(
        i32, [10, 4], memory_space=builtin.IntegerAttr(2, i32)
    )

    memory_space_cast = MemorySpaceCastOp(memref_ssa_value, res_type)

    assert memory_space_cast.source is memref_ssa_value
    assert memory_space_cast.dest.type is res_type

    with pytest.raises(
        VerifyException,
        match="Expected source and destination to have the same element type.",
    ):
        MemorySpaceCastOp(memref_ssa_value, res_type_wrong_type).verify()

    with pytest.raises(
        VerifyException, match="Expected source and destination to have the same shape."
    ):
        MemorySpaceCastOp(memref_ssa_value, res_type_wrong_shape).verify()

    # Test helper function
    dest_memory_space = builtin.IntegerAttr(2, i32)
    memory_space_cast = MemorySpaceCastOp.from_type_and_target_space(
        memref_ssa_value, i32_memref_type, dest_memory_space
    )

    assert memory_space_cast.source is memref_ssa_value
    assert isinstance(memory_space_cast.dest.type, MemRefType)
    assert memory_space_cast.dest.type.memory_space is dest_memory_space


def test_dma_start():
    src_type = MemRefType(i64, [4, 512], memory_space=IntAttr(1))
    dest_type = MemRefType(i64, [4 * 512], memory_space=IntAttr(2))

    tag_type = MemRefType(i32, [4])

    src = create_ssa_value(src_type)
    dest = create_ssa_value(dest_type)
    tag = create_ssa_value(tag_type)

    index = create_ssa_value(IndexType())
    num_elements = create_ssa_value(IndexType())

    dma_start = DmaStartOp.get(
        src, [index, index], dest, [index], num_elements, tag, [index]
    )

    dma_start.verify()

    # check that src index count is verified
    with pytest.raises(VerifyException, match="Expected 2 source indices"):
        DmaStartOp.get(
            src, [index, index, index], dest, [index], num_elements, tag, [index]
        ).verify()

    # check that dest index count is verified
    with pytest.raises(VerifyException, match="Expected 1 dest indices"):
        DmaStartOp.get(
            src, [index, index], dest, [], num_elements, tag, [index]
        ).verify()

    # check that tag index count is verified
    with pytest.raises(VerifyException, match="Expected 1 tag indices"):
        DmaStartOp.get(
            src, [index, index], dest, [index], num_elements, tag, [index, index]
        ).verify()

    # check that tag index count is verified
    with pytest.raises(VerifyException, match="different memory spaces"):
        DmaStartOp.get(
            src, [index, index], src, [index, index], num_elements, tag, [index]
        ).verify()

    # check that tag element type is verified
    with pytest.raises(VerifyException, match="Expected tag to be a memref of i32"):
        new_tag = create_ssa_value(src_type)

        DmaStartOp.get(
            src,
            [index, index],
            dest,
            [index],
            num_elements,
            new_tag,
            [index, index],
        ).verify()


def test_memref_dma_wait():
    tag_type = MemRefType(i32, [4])
    tag = create_ssa_value(tag_type)
    index = create_ssa_value(IndexType())
    num_elements = create_ssa_value(IndexType())

    dma_wait = DmaWaitOp.get(tag, [index], num_elements)

    dma_wait.verify()

    # check that tag index count is verified
    with pytest.raises(
        VerifyException, match="Expected 1 tag indices because of shape of tag memref"
    ):
        DmaWaitOp.get(tag, [index, index], num_elements).verify()

    # check that tag element type is verified
    with pytest.raises(VerifyException, match="Expected tag to be a memref of i32"):
        wrong_tag_type = MemRefType(i64, [4])
        wrong_tag = create_ssa_value(wrong_tag_type)

        DmaWaitOp.get(wrong_tag, [index], num_elements).verify()


def test_memref_copy():
    i32type4 = MemRefType(i32, [4])
    i32type3 = MemRefType(i32, [3])
    i64type4 = MemRefType(i64, [4])
    source = create_ssa_value(i32type4)
    destination = create_ssa_value(i32type4)

    copy = CopyOp(source, destination)

    copy.verify()
    assert isinstance(copy, CopyOp)
    assert copy.source.type == i32type4
    assert copy.destination.type == i32type4

    destination = create_ssa_value(i32type3)

    copy = CopyOp(source, destination)

    with pytest.raises(
        VerifyException, match="Expected source and destination to have the same shape."
    ):
        copy.verify()

    destination = create_ssa_value(i64type4)

    copy = CopyOp(source, destination)

    with pytest.raises(
        VerifyException,
        match="Expected source and destination to have the same element type.",
    ):
        copy.verify()


def test_get_strides():
    strided = StridedLayoutAttr((24, 4, 1), 5)
    affine = AffineMapAttr(AffineMap.identity(3))

    assert tuple(strided.get_strides()) == (24, 4, 1)
    assert affine.get_strides() is None

    t_none = MemRefType(i32, (2, 3, 4))
    assert (strides := t_none.get_strides())
    assert tuple(strides) == (12, 4, 1)

    t_id = MemRefType(i32, (2, 3, 4), strided)
    assert (strides := t_id.get_strides())
    assert tuple(strides) == (24, 4, 1)
