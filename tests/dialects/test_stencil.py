import pytest

from xdsl.builder import Builder
from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerType,
    MemRefType,
    bf16,
    f16,
    f32,
    f64,
    f80,
    f128,
    i32,
    i64,
)
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    BufferOp,
    CastOp,
    DynAccessOp,
    ExternalLoadOp,
    ExternalStoreOp,
    FieldType,
    IndexAttr,
    IndexOp,
    LoadOp,
    ResultType,
    ReturnOp,
    StencilBoundsAttr,
    StoreOp,
    StoreResultOp,
    TempType,
)
from xdsl.ir import Attribute, Block, SSAValue
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.test_value import create_ssa_value


def test_stencilboundsattr_verify():
    with pytest.raises(
        VerifyException,
        match=(
            "Incoherent stencil bounds: lower and upper bounds must have the same"
            " dimensionality."
        ),
    ):
        StencilBoundsAttr.new([IndexAttr.get(1), IndexAttr.get(2, 2)])

    with pytest.raises(
        VerifyException,
        match=(
            "Incoherent stencil bounds: upper bound must be strictly greater than"
            " lower bound."
        ),
    ):
        StencilBoundsAttr.new([IndexAttr.get(2, 2), IndexAttr.get(2, 2)])


def test_stencil_return_single_float():
    float_val1 = create_ssa_value(FloatAttr(4.0, f32))
    return_op = ReturnOp.get([float_val1])

    assert return_op.arg[0] is float_val1


def test_stencil_return_multiple_floats():
    float_val1 = create_ssa_value(FloatAttr(4.0, f32))
    float_val2 = create_ssa_value(FloatAttr(5.0, f32))
    float_val3 = create_ssa_value(FloatAttr(6.0, f32))

    return_op = ReturnOp.get([float_val1, float_val2, float_val3])

    assert return_op.arg[0] is float_val1
    assert return_op.arg[1] is float_val2
    assert return_op.arg[2] is float_val3


def test_stencil_return_single_ResultType():
    result_type_val1 = create_ssa_value(ResultType(f32))
    return_op = ReturnOp.get([result_type_val1])

    assert return_op.arg[0] is result_type_val1


def test_stencil_return_multiple_ResultType():
    result_type_val1 = create_ssa_value(ResultType(f32))
    result_type_val2 = create_ssa_value(ResultType(f32))
    result_type_val3 = create_ssa_value(ResultType(f32))

    return_op = ReturnOp.get([result_type_val1, result_type_val2, result_type_val3])

    assert return_op.arg[0] is result_type_val1
    assert return_op.arg[1] is result_type_val2
    assert return_op.arg[2] is result_type_val3


def test_stencil_cast_op_verifier():
    field_type = FieldType(3, f32)
    field = create_ssa_value(field_type)

    # check that correct op verifies correctly
    cast = CastOp.get(field, StencilBoundsAttr(((-2, 100), (-2, 100), (-2, 100))))
    cast.verify()

    # check that output has same dims as input and lb, ub
    with pytest.raises(
        VerifyException, match="Input and output types must have the same rank"
    ):
        cast = CastOp.get(
            field,
            StencilBoundsAttr(((-2, 100), (-2, 100), (-2, 100))),
            FieldType(((-2, 102), (-2, 102)), f32),
        )
        cast.verify()

    # check that input and output have same element type
    with pytest.raises(
        VerifyException,
        match="Input and output fields must have the same element types",
    ):
        cast = CastOp.get(
            field,
            StencilBoundsAttr(((-2, 100), (-2, 100), (-2, 100))),
            FieldType(((-2, 102), (-2, 102), (-2, 102)), f64),
        )
        cast.verify()

    # check that non-dynamic input verifies
    non_dyn_field = create_ssa_value(FieldType(((-2, 102), (-2, 102), (-2, 102)), f32))
    cast = CastOp.get(
        non_dyn_field,
        StencilBoundsAttr(((-2, 100), (-2, 100), (-2, 100))),
        FieldType(((-2, 102), (-2, 102), (-2, 102)), f32),
    )
    cast.verify()

    with pytest.raises(
        VerifyException,
        match="If input shape is not dynamic, it must be the same as output",
    ):
        cast = CastOp.get(
            non_dyn_field,
            StencilBoundsAttr(((-2, 100), (-2, 100), (-2, 101))),
            FieldType(((-2, 102), (-2, 102), (-3, 103)), f32),
        )
        cast.verify()


def test_cast_op_constructor():
    field = create_ssa_value(FieldType(3, f32))

    cast = CastOp.get(
        field,
        StencilBoundsAttr(((-2, 100), (-3, 100), (-4, 0))),
    )

    assert cast.result.type == FieldType(((-2, 100), (-3, 100), (-4, 0)), f32)


def test_stencil_apply():
    result_type_val1 = create_ssa_value(ResultType(f32))

    stencil_temptype = TempType(2, f32)
    apply_op = ApplyOp.get([result_type_val1], Block([]), [stencil_temptype])

    assert len(apply_op.args) == 1
    assert len(apply_op.res) == 1
    assert isinstance(apply_op.res[0].type, TempType)
    assert apply_op.get_rank() == 2


def test_stencil_apply_no_args():
    stencil_temptype = TempType(1, f32)
    apply_op = ApplyOp.get([], Block([]), [stencil_temptype, stencil_temptype])

    assert len(apply_op.args) == 0
    assert len(apply_op.res) == 2
    assert isinstance(apply_op.res[0].type, TempType)
    assert apply_op.get_rank() == 1


def test_stencil_apply_no_results():
    # Should error if there are no results expected
    with pytest.raises(AssertionError):
        ApplyOp.get([], Block([]), [])


@pytest.mark.parametrize(
    "indices",
    [
        ([1]),
        ([1, 2]),
        ([1, 2, 3]),
        (
            [
                IntAttr(1),
                IntAttr(2),
                IntAttr(3),
            ]
        ),
    ],
)
def test_create_index_attr_from_int_list(indices: list[int | IntAttr]):
    stencil_index_attr = IndexAttr.get(*indices)
    expected_array_attr = ArrayAttr(
        [(IntAttr(idx) if isinstance(idx, int) else idx) for idx in indices]
    )

    assert stencil_index_attr.array == expected_array_attr


def test_create_index_attr_from_list_edge_case1():
    with pytest.raises(
        VerifyException, match="Expected 1 to 3 indexes for stencil.index, got 0."
    ):
        IndexAttr.get()


def test_create_index_attr_from_list_edge_case2():
    with pytest.raises(
        VerifyException, match="Expected 1 to 3 indexes for stencil.index, got 4."
    ):
        IndexAttr.get(*[1] * 4)


@pytest.mark.parametrize(
    "indices",
    [([1]), ([1, 2]), ([1, 2, 3])],
)
def test_index_attr_neg(indices: list[int]):
    stencil_index_attr = IndexAttr.get(*indices)
    stencil_index_attr_neg = -stencil_index_attr
    expected_array_attr = ArrayAttr([(IntAttr(-idx)) for idx in indices])

    assert stencil_index_attr_neg.array == expected_array_attr


@pytest.mark.parametrize(
    "indices1, indices2",
    [([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])],
)
def test_index_attr_add(indices1: list[int], indices2: list[int]):
    stencil_index_attr1 = IndexAttr.get(*indices1)
    stencil_index_attr2 = IndexAttr.get(*indices2)

    stencil_index_attr_add = stencil_index_attr1 + stencil_index_attr2
    expected_array_attr = ArrayAttr(
        [(IntAttr(idx1 + idx2)) for idx1, idx2 in zip(indices1, indices2)]
    )

    assert stencil_index_attr_add.array == expected_array_attr


@pytest.mark.parametrize(
    "indices1, indices2",
    [([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])],
)
def test_index_attr_sub(indices1: list[int], indices2: list[int]):
    stencil_index_attr1 = IndexAttr.get(*indices1)
    stencil_index_attr2 = IndexAttr.get(*indices2)

    stencil_index_attr_sub = stencil_index_attr1 - stencil_index_attr2
    expected_array_attr = ArrayAttr(
        [(IntAttr(idx1 - idx2)) for idx1, idx2 in zip(indices1, indices2)]
    )

    assert stencil_index_attr_sub.array == expected_array_attr


@pytest.mark.parametrize(
    "indices1, indices2",
    [([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])],
)
def test_index_attr_min(indices1: list[int], indices2: list[int]):
    stencil_index_attr1 = IndexAttr.get(*indices1)
    stencil_index_attr2 = IndexAttr.get(*indices2)

    stencil_index_attr_min = IndexAttr.min(stencil_index_attr1, stencil_index_attr2)
    expected_array_attr = ArrayAttr(
        [(IntAttr(min(idx1, idx2))) for idx1, idx2 in zip(indices1, indices2)]
    )

    assert stencil_index_attr_min.array == expected_array_attr


@pytest.mark.parametrize(
    "indices1, indices2",
    [([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])],
)
def test_index_attr_max(indices1: list[int], indices2: list[int]):
    stencil_index_attr1 = IndexAttr.get(*indices1)
    stencil_index_attr2 = IndexAttr.get(*indices2)

    stencil_index_attr_max = IndexAttr.max(stencil_index_attr1, stencil_index_attr2)
    expected_array_attr = ArrayAttr(
        [(IntAttr(max(idx1, idx2))) for idx1, idx2 in zip(indices1, indices2)]
    )

    assert stencil_index_attr_max.array == expected_array_attr


@pytest.mark.parametrize(
    "indices",
    [((1,)), ((1, 2)), ((1, 2, 3))],
)
def test_index_attr_iter(indices: tuple[int, ...]):
    stencil_index_attr = IndexAttr.get(*indices)

    assert tuple(stencil_index_attr) == indices


@pytest.mark.parametrize("indices", [([1]), ([1, 2]), ([1, 2, 3])])
def test_index_attr_indices_length(indices: list[int]):
    stencil_index_attr = IndexAttr.get(*indices)
    stencil_index_attr_iter = iter(stencil_index_attr)

    for idx in indices:
        assert idx == next(stencil_index_attr_iter)


@pytest.mark.parametrize(
    "attr, bounds",
    [
        (i32, ((0, 64), (0, 64))),
        (
            i64,
            ((0, 32), (0, 32), (0, 32)),
        ),
    ],
)
def test_stencil_fieldtype_constructor_with_ArrayAttr(
    attr: IntegerType, bounds: tuple[tuple[int, int], ...]
):
    stencil_fieldtype = FieldType(bounds, attr)

    assert stencil_fieldtype.element_type == attr
    assert stencil_fieldtype.get_num_dims() == len(bounds)
    assert isinstance(stencil_fieldtype.bounds, StencilBoundsAttr)
    assert (
        tuple(zip(stencil_fieldtype.bounds.lb, stencil_fieldtype.bounds.ub)) == bounds
    )


@pytest.mark.parametrize(
    "attr, bounds",
    [
        (i32, ((0, 1), (0, 2))),
        (i32, ((0, 1), (0, 1), (0, 3))),
        (i64, ((0, 1), (0, 1), (0, 3))),
    ],
)
def test_stencil_fieldtype_constructor(
    attr: IntegerType, bounds: tuple[tuple[int, int], ...]
):
    stencil_fieldtype = FieldType(bounds, attr)

    assert stencil_fieldtype.element_type == attr
    assert stencil_fieldtype.get_num_dims() == len(bounds)
    assert isinstance(stencil_fieldtype.bounds, StencilBoundsAttr)
    assert (
        tuple(zip(stencil_fieldtype.bounds.lb, stencil_fieldtype.bounds.ub)) == bounds
    )


@pytest.mark.parametrize(
    "attr, bounds",
    [
        (i32, []),
        (i64, []),
    ],
)
def test_stencil_fieldtype_constructor_empty_list(
    attr: IntegerType, bounds: list[tuple[int, int]]
):
    with pytest.raises(
        VerifyException, match="Expected 1 to 3 indexes for stencil.index, got 0."
    ):
        FieldType(bounds, attr)


def test_stencil_load():
    field_type = FieldType([(0, 1), (0, 1)], f32)
    result_type_val1 = create_ssa_value(field_type)

    load = LoadOp.get(result_type_val1)

    assert isinstance(load_field_type := load.field.type, FieldType)
    assert load_field_type == field_type
    assert len(load_field_type.get_shape()) == 2
    assert isinstance(load_field_type.bounds, StencilBoundsAttr)
    assert isa(load.res.type, TempType[Attribute])
    assert isa(load.res.type.bounds, IntAttr)
    assert load.res.type.bounds.data == 2


def test_stencil_load_bounds():
    field_type = FieldType([(0, 1), (0, 1)], f32)
    result_type_val1 = create_ssa_value(field_type)

    lb = IndexAttr.get(1, 1)
    ub = IndexAttr.get(64, 64)

    load = LoadOp.get(result_type_val1, lb, ub)

    assert isa(load.res.type, TempType[Attribute])
    assert isinstance(load.res.type.bounds, StencilBoundsAttr)
    assert isinstance(load.res.type.bounds.lb, IndexAttr)
    assert isinstance(load.res.type.bounds.ub, IndexAttr)
    assert len(load.res.type.bounds.lb) == 2
    assert load.res.type.bounds.lb == lb
    assert len(load.res.type.bounds.ub) == 2
    assert load.res.type.bounds.ub == ub


@pytest.mark.parametrize(
    "attr, dims",
    [
        (i32, ((0, 64), (0, 64))),
        (
            i64,
            ((0, 32), (0, 32), (0, 32)),
        ),
    ],
)
def test_stencil_temptype_constructor_with_ArrayAttr(
    attr: IntegerType, dims: tuple[tuple[int, int], ...]
):
    stencil_temptype = TempType(dims, attr)

    assert isinstance(stencil_temptype, TempType)
    assert stencil_temptype.element_type == attr
    assert stencil_temptype.get_num_dims() == len(dims)
    assert isinstance(stencil_temptype.bounds, StencilBoundsAttr)
    assert tuple(zip(stencil_temptype.bounds.lb, stencil_temptype.bounds.ub)) == dims


@pytest.mark.parametrize(
    "attr, dims",
    [
        (i32, ((0, 1), (0, 2))),
        (i32, ((0, 1), (0, 1), (0, 3))),
        (i64, ((0, 1), (0, 1), (0, 3))),
    ],
)
def test_stencil_temptype_constructor(
    attr: IntegerType, dims: tuple[tuple[int, int], ...]
):
    stencil_temptype = TempType(dims, attr)

    assert isinstance(stencil_temptype, TempType)
    assert stencil_temptype.element_type == attr
    assert stencil_temptype.get_num_dims() == len(dims)
    assert isinstance(stencil_temptype.bounds, StencilBoundsAttr)
    assert tuple(zip(stencil_temptype.bounds.lb, stencil_temptype.bounds.ub)) == dims


@pytest.mark.parametrize(
    "attr, dims",
    [
        (i32, []),
        (i64, []),
    ],
)
def test_stencil_temptype_constructor_empty_list(
    attr: IntegerType, dims: list[tuple[int, int]]
):
    with pytest.raises(
        VerifyException, match="Expected 1 to 3 indexes for stencil.index, got 0."
    ):
        TempType(dims, attr)


@pytest.mark.parametrize(
    "float_type",
    [(bf16), (f16), (f32), (f64), (f80), (f128)],
)
def test_stencil_resulttype(float_type: AnyFloat):
    stencil_resulttype = ResultType(float_type)

    assert isinstance(stencil_resulttype, ResultType)
    assert stencil_resulttype.elem == float_type


def test_stencil_store():
    temp_type = TempType([(0, 5), (0, 5)], f32)
    temp_type_ssa_val = create_ssa_value(temp_type)

    field_type = FieldType([(0, 2), (0, 2)], f32)
    field_type_ssa_val = create_ssa_value(field_type)

    lb = IndexAttr.get(1, 1)
    ub = IndexAttr.get(64, 64)
    bounds = StencilBoundsAttr.new((lb, ub))

    store = StoreOp.get(temp_type_ssa_val, field_type_ssa_val, bounds)

    assert isinstance(store, StoreOp)
    assert isinstance(store_field_type := store.field.type, FieldType)
    assert store_field_type == field_type
    assert isinstance(store_temp_type := store.temp.type, TempType)
    assert store_temp_type == temp_type
    assert len(store_field_type.get_shape()) == 2
    assert len(store_temp_type.get_shape()) == 2
    assert store.bounds is bounds


def test_stencil_index():
    dim = IntAttr(10)
    offset = IndexAttr.get(1)

    index = IndexOp.build(
        attributes={
            "dim": dim,
            "offset": offset,
        },
        result_types=[IndexType()],
    )

    assert isinstance(index, IndexOp)
    assert index.dim is dim
    assert index.offset is offset


def test_stencil_access():
    temp_type = TempType([(0, 5), (0, 5)], f32)
    temp_type_ssa_val = create_ssa_value(temp_type)

    offset = [1, 1]
    offset_index_attr = IndexAttr.get(*offset)

    access = AccessOp.get(temp_type_ssa_val, offset)

    assert isinstance(access, AccessOp)
    assert access.offset == offset_index_attr
    assert access.temp.type == temp_type


def test_stencil_dyn_access():
    temp_type = TempType([(0, 5), (0, 5)], f32)
    temp_type_ssa_val = create_ssa_value(temp_type)

    lb = IndexAttr.get(0, 0)
    ub = IndexAttr.get(1, 1)
    offset = (
        create_ssa_value(builtin.IndexType()),
        create_ssa_value(builtin.IndexType()),
    )

    dyn_access = DynAccessOp(temp_type_ssa_val, offset, lb, ub)

    assert dyn_access.offset == offset
    assert dyn_access.temp is temp_type_ssa_val
    assert dyn_access.lb is lb
    assert dyn_access.ub is ub


def test_stencil_access_offset_mapping():
    temp_type = TempType([(0, 5), (0, 5)], f32)
    temp_type_ssa_val = create_ssa_value(temp_type)

    offset = [1, 1]
    offset_index_attr = IndexAttr.get(*offset)

    offset_mapping = [0, 1]
    offset_mapping_attr = IndexAttr.get(*offset_mapping)

    access = AccessOp.get(temp_type_ssa_val, offset, offset_mapping)

    assert isinstance(access, AccessOp)
    assert access.offset == offset_index_attr
    assert access.temp.type == temp_type
    assert access.offset_mapping is not None
    assert access.offset_mapping == offset_mapping_attr


def test_store_result():
    elem = IndexAttr.get(1)
    elem_ssa_val = create_ssa_value(elem)
    result_type = ResultType(f32)

    store_result = StoreResultOp.build(
        operands=[elem_ssa_val], result_types=[result_type]
    )

    assert isinstance(store_result, StoreResultOp)
    assert store_result.arg == elem_ssa_val
    assert store_result.res.type == result_type


def test_external_load():
    memref = create_ssa_value(MemRefType(f32, [5]))
    field_type = FieldType((5), f32)

    external_load = ExternalLoadOp.get(memref, field_type)

    assert isinstance(external_load, ExternalLoadOp)
    assert external_load.field == memref
    assert external_load.result.type == field_type


def test_external_store():
    field = create_ssa_value(FieldType(5, f32))
    memref = create_ssa_value(MemRefType(f32, [5]))

    external_store = ExternalStoreOp.build(operands=[field, memref])

    assert isinstance(external_store, ExternalStoreOp)
    assert external_store.field == memref
    assert external_store.temp == field


def test_buffer():
    temp = create_ssa_value(TempType((5), f32))
    res_type = TempType((5), f32)

    buffer = BufferOp.build(operands=[temp], result_types=[res_type])

    assert isinstance(buffer, BufferOp)
    assert buffer.temp == temp
    assert buffer.res.type == res_type


def test_access_patterns():
    typ = TempType((5), f32)
    temp = create_ssa_value(typ)

    @Builder.implicit_region((typ, typ))
    def apply_op_region(args: tuple[SSAValue, ...]):
        t0, t1 = args
        for x in (-1, 1):
            AccessOp.get(t0, (x, 0), (1, 0))
        for y in (-1, 1):
            AccessOp.get(t0, (0, y), (1, 0))

        AccessOp.get(t1, (1, 1), (1, 0))
        AccessOp.get(t1, (-1, -1), (1, 0))

    apply = ApplyOp.get((temp, temp), apply_op_region.detach_block(0), [typ])

    t0_acc, t1_acc = tuple(apply.get_accesses())

    assert t0_acc.visual_pattern() == " X \nXOX\n X "
    assert t1_acc.visual_pattern() == "X  \n O \n  X"

    assert not t0_acc.is_diagonal
    assert t1_acc.is_diagonal

    assert len(tuple(t1_acc.get_diagonals())) == 2
    assert t0_acc.max_distance() == 1
    assert t1_acc.max_distance() == 1
