import pytest

from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    StridedLayoutAttr,
    i32,
    i64,
)
from xdsl.dialects.memref import (
    Alloc,
    Alloca,
    Cast,
    CopyOp,
    Dealloc,
    DmaStartOp,
    DmaWaitOp,
    ExtractAlignedPointerAsIndexOp,
    Load,
    MemRefType,
    Store,
    Subview,
    UnrankedMemrefType,
)
from xdsl.ir import Attribute, OpResult
from xdsl.ir.core import BlockArgument
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.test_value import TestSSAValue


def test_memreftype():
    mem1 = MemRefType.from_element_type_and_shape(i32, [1])

    assert mem1.get_num_dims() == 1
    assert mem1.get_shape() == (1,)
    assert mem1.element_type is i32

    mem2 = MemRefType.from_element_type_and_shape(i32, [3, 3, 3])

    assert mem2.get_num_dims() == 3
    assert mem2.get_shape() == (3, 3, 3)
    assert mem2.element_type is i32

    my_i32 = IntegerType(32)
    mem3 = MemRefType.from_params(my_i32)

    assert mem3.get_num_dims() == 1
    assert mem3.get_shape() == (1,)
    assert mem3.element_type is my_i32


def test_memref_load_i32():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = TestSSAValue(i32_memref_type)
    load = Load.get(memref_ssa_value, [])

    assert load.memref is memref_ssa_value
    assert load.indices == ()
    assert load.res.type is i32


def test_memref_load_i32_with_dimensions():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = TestSSAValue(i32_memref_type)
    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())
    load = Load.get(memref_ssa_value, [index1, index2])

    assert load.memref is memref_ssa_value
    assert load.indices[0] is index1
    assert load.indices[1] is index2
    assert load.res.type is i32


def test_memref_store_i32():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [1])
    memref_ssa_value = TestSSAValue(i32_memref_type)
    i32_ssa_value = TestSSAValue(i32)
    store = Store.get(i32_ssa_value, memref_ssa_value, [])

    assert store.memref is memref_ssa_value
    assert store.indices == ()
    assert store.value is i32_ssa_value


def test_memref_store_i32_with_dimensions():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [2, 3])
    memref_ssa_value = TestSSAValue(i32_memref_type)
    i32_ssa_value = TestSSAValue(i32)
    index1 = TestSSAValue(IndexType())
    index2 = TestSSAValue(IndexType())
    store = Store.get(i32_ssa_value, memref_ssa_value, [index1, index2])

    assert store.memref is memref_ssa_value
    assert store.indices[0] is index1
    assert store.indices[1] is index2
    assert store.value is i32_ssa_value


def test_memref_alloc():
    my_i32 = IntegerType(32)
    alloc0 = Alloc.get(my_i32, 64, [3, 1, 2])
    alloc1 = Alloc.get(my_i32, 64)

    assert alloc0.dynamic_sizes == ()
    assert type(alloc0.results[0]) is OpResult
    assert type(alloc0.results[0].type) is MemRefType
    assert alloc0.results[0].type.get_shape() == (3, 1, 2)
    assert type(alloc1.results[0]) is OpResult
    assert type(alloc1.results[0].type) is MemRefType
    assert alloc1.results[0].type.get_shape() == (1,)


def test_memref_alloca():
    my_i32 = IntegerType(32)
    alloc0 = Alloca.get(my_i32, 64, [3, 1, 2])
    alloc1 = Alloca.get(my_i32, 64)

    assert type(alloc0.results[0]) is OpResult
    assert type(alloc0.results[0].type) is MemRefType
    assert alloc0.results[0].type.get_shape() == (3, 1, 2)
    assert type(alloc1.results[0]) is OpResult
    assert type(alloc1.results[0].type) is MemRefType
    assert alloc1.results[0].type.get_shape() == (1,)


def test_memref_dealloc():
    my_i32 = IntegerType(32)
    alloc0 = Alloc.get(my_i32, 64, [3, 1, 2])
    dealloc0 = Dealloc.get(alloc0)

    assert type(dealloc0.memref) is OpResult


def test_memref_dim():
    idx = arith.Constant.from_int_and_width(1, IndexType())
    alloc0 = Alloc.get(i32, 64, [3, 1, 2])
    dim_1 = memref.Dim.from_source_and_index(alloc0, idx)

    assert dim_1.source is alloc0.memref
    assert dim_1.index is idx.result
    assert isinstance(dim_1.result.type, IndexType)


def test_memref_rank():
    alloc0 = Alloc.get(i32, 64, [3, 1, 2])
    dim_1 = memref.Rank.from_memref(alloc0)

    assert dim_1.source is alloc0.memref
    assert isinstance(dim_1.rank.type, IndexType)


def test_memref_ExtractAlignedPointerAsIndexOp():
    ref = Alloc.get(i32, 64, [64, 64, 64])
    ptr = ExtractAlignedPointerAsIndexOp.get(ref)

    assert ptr.aligned_pointer.type == IndexType()
    assert ptr.source == ref.memref


def test_memref_matmul_verify():
    memref_f64_rank2 = memref.MemRefType.from_element_type_and_shape(
        builtin.f64, [-1, -1]
    )

    @builtin.ModuleOp
    @Builder.implicit_region
    def module():
        @Builder.implicit_region((memref_f64_rank2, memref_f64_rank2))
        def matmul(args: tuple[BlockArgument, ...]) -> None:
            a, b = args

            lit0 = arith.Constant.from_int_and_width(0, builtin.IndexType())
            lit1 = arith.Constant.from_int_and_width(1, builtin.IndexType())
            dim_a0 = memref.Dim.from_source_and_index(a, lit0)
            dim_a1 = memref.Dim.from_source_and_index(a, lit1)
            dim_b0 = memref.Dim.from_source_and_index(b, lit0)
            dim_b1 = memref.Dim.from_source_and_index(b, lit1)
            out = memref.Alloca.get(builtin.f64, 0, [-1, -1], [dim_a0, dim_b1])
            # TODO: assert dim_a0 == dim_b1
            lit0_f = arith.Constant.from_float_and_width(0.0, builtin.f64)

            @Builder.implicit_region((builtin.IndexType(),))
            def outer_loop(args: tuple[BlockArgument, ...]):
                (i,) = args

                # outer loop start, loop_var = i

                @Builder.implicit_region((builtin.IndexType(),))
                def mid_loop(args: tuple[BlockArgument, ...]):
                    (j,) = args
                    # mid loop start, loop_var = j
                    memref.Store.get(lit0_f, out, [i, j])

                    @Builder.implicit_region((builtin.IndexType(),))
                    def inner_loop(args: tuple[BlockArgument, ...]):
                        (k,) = args
                        # inner loop, loop_var = k
                        elem_a_i_k = memref.Load.get(a, [i, k])
                        elem_b_k_j = memref.Load.get(b, [k, j])
                        mul = arith.Mulf(elem_a_i_k, elem_b_k_j)
                        out_i_j = memref.Load.get(out, [i, j])
                        new_out_val = arith.Addf(out_i_j, mul)
                        memref.Store.get(new_out_val, out, [i, j])
                        scf.Yield.get()

                    scf.For.get(lit0, dim_a1, lit1, [], inner_loop)
                    scf.Yield.get()

                scf.For.get(lit0, dim_b0, lit1, [], mid_loop)
                scf.Yield.get()

            scf.For.get(lit0, dim_a0, lit1, [], outer_loop)

            func.Return(out)

        func.FuncOp(
            "matmul",
            ((memref_f64_rank2, memref_f64_rank2), (memref_f64_rank2,)),
            matmul,
        )

    # check that it verifies correctly
    module.verify()


def test_memref_subview():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [10, 2])
    memref_ssa_value = TestSSAValue(i32_memref_type)

    res_memref_type = MemRefType.from_element_type_and_shape(i32, [1, 1])

    offset_arg1 = Constant.from_attr(IntegerAttr.from_int_and_width(0, 64), i64)
    offset_arg2 = Constant.from_attr(IntegerAttr.from_int_and_width(0, 64), i64)

    size_arg1 = Constant.from_attr(IntegerAttr.from_int_and_width(1, 64), i64)
    size_arg2 = Constant.from_attr(IntegerAttr.from_int_and_width(1, 64), i64)

    stride_arg1 = Constant.from_attr(IntegerAttr.from_int_and_width(1, 64), i64)
    stride_arg2 = Constant.from_attr(IntegerAttr.from_int_and_width(1, 64), i64)

    operand_segment_sizes = ArrayAttr([IntAttr(1), IntAttr(2), IntAttr(2), IntAttr(2)])

    static_offsets = DenseArrayBase.from_list(i64, [0, 0])
    static_sizes = DenseArrayBase.from_list(i64, [1, 1])
    static_strides = DenseArrayBase.from_list(i64, [1, 1])

    subview = Subview.build(
        operands=[
            memref_ssa_value,
            [offset_arg1, offset_arg2],
            [size_arg1, size_arg2],
            [stride_arg1, stride_arg2],
        ],
        attributes={
            "operand_segment_sizes": operand_segment_sizes,
            "static_offsets": static_offsets,
            "static_sizes": static_sizes,
            "static_strides": static_strides,
        },
        result_types=[res_memref_type],
    )

    assert subview.source is memref_ssa_value
    assert subview.offsets == (offset_arg1.result, offset_arg2.result)
    assert subview.sizes == (size_arg1.result, size_arg2.result)
    assert subview.strides == (stride_arg1.result, stride_arg2.result)
    assert subview.static_offsets is static_offsets
    assert subview.static_sizes is static_sizes
    assert subview.static_strides is static_strides
    assert subview.result.type is res_memref_type


def test_memref_subview_constant_parameters():
    alloc = Alloc.get(i32, 8, [10, 10, 10])
    assert isa(alloc.memref.type, MemRefType[Attribute])

    subview = Subview.from_static_parameters(
        alloc, alloc.memref.type, [2, 2, 2], [2, 2, 2], [3, 3, 3]
    )

    assert isinstance(subview, Subview)
    assert isinstance(subview.result.type, MemRefType)
    assert isinstance(subview.result.type.layout, StridedLayoutAttr)
    assert isa(subview.result.type.layout.strides, ArrayAttr[IntAttr])
    out_strides = [i.data for i in subview.result.type.layout.strides.data]
    assert out_strides == [300, 30, 3]
    assert isinstance(subview.result.type.layout.offset, IntAttr)
    assert subview.result.type.layout.offset.data == 222


def test_memref_cast():
    i32_memref_type = MemRefType.from_element_type_and_shape(i32, [10, 2])
    memref_ssa_value = TestSSAValue(i32_memref_type)

    res_type = UnrankedMemrefType.from_type(i32)

    cast = Cast.get(memref_ssa_value, res_type)

    assert cast.source is memref_ssa_value
    assert cast.dest.type is res_type


def test_dma_start():
    src_type = MemRefType.from_element_type_and_shape(
        i64, [4, 512], memory_space=IntAttr(1)
    )
    dest_type = MemRefType.from_element_type_and_shape(
        i64, [4 * 512], memory_space=IntAttr(2)
    )

    tag_type = MemRefType.from_element_type_and_shape(i32, [4])

    src = TestSSAValue(src_type)
    dest = TestSSAValue(dest_type)
    tag = TestSSAValue(tag_type)

    index = TestSSAValue(IndexType())
    num_elements = TestSSAValue(IndexType())

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
        new_tag = TestSSAValue(src_type)

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
    tag_type = MemRefType.from_element_type_and_shape(i32, [4])
    tag = TestSSAValue(tag_type)
    index = TestSSAValue(IndexType())
    num_elements = TestSSAValue(IndexType())

    dma_wait = DmaWaitOp.get(tag, [index], num_elements)

    dma_wait.verify()

    # check that tag index count is verified
    with pytest.raises(
        VerifyException, match="Expected 1 tag indices because of shape of tag memref"
    ):
        DmaWaitOp.get(tag, [index, index], num_elements).verify()

    # check that tag element type is verified
    with pytest.raises(VerifyException, match="Expected tag to be a memref of i32"):
        wrong_tag_type = MemRefType.from_element_type_and_shape(i64, [4])
        wrong_tag = TestSSAValue(wrong_tag_type)

        DmaWaitOp.get(wrong_tag, [index], num_elements).verify()


def test_memref_copy():
    i32type4 = MemRefType.from_element_type_and_shape(i32, [4])
    i32type3 = MemRefType.from_element_type_and_shape(i32, [3])
    i64type4 = MemRefType.from_element_type_and_shape(i64, [4])
    source = TestSSAValue(i32type4)
    destination = TestSSAValue(i32type4)

    copy = CopyOp(source, destination)

    copy.verify()
    assert isinstance(copy, CopyOp)
    assert copy.source.type == i32type4
    assert copy.destination.type == i32type4

    destination = TestSSAValue(i32type3)

    copy = CopyOp(source, destination)

    with pytest.raises(
        VerifyException, match="Expected source and destination to have the same shape."
    ):
        copy.verify()

    destination = TestSSAValue(i64type4)

    copy = CopyOp(source, destination)

    with pytest.raises(
        VerifyException,
        match="Expected source and destination to have the same element type.",
    ):
        copy.verify()
