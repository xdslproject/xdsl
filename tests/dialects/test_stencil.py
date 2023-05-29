import pytest

from xdsl.dialects.builtin import (
    AnyFloat,
    FloatAttr,
    IntAttr,
    bf16,
    f16,
    f32,
    f64,
    f80,
    f128,
    i32,
    i64,
    IntegerType,
    ArrayAttr,
)
from xdsl.dialects.experimental.stencil import (
    ReturnOp,
    ResultType,
    ApplyOp,
    StencilBoundsAttr,
    TempType,
    LoadOp,
    FieldType,
    IndexAttr,
)
from xdsl.dialects.stencil import CastOp
from xdsl.ir import Attribute, Block
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.test_value import TestSSAValue


def test_stencil_return_single_float():
    float_val1 = TestSSAValue(FloatAttr(4.0, f32))
    return_op = ReturnOp.get([float_val1])

    assert return_op.arg[0] is float_val1


def test_stencil_return_multiple_floats():
    float_val1 = TestSSAValue(FloatAttr(4.0, f32))
    float_val2 = TestSSAValue(FloatAttr(5.0, f32))
    float_val3 = TestSSAValue(FloatAttr(6.0, f32))

    return_op = ReturnOp.get([float_val1, float_val2, float_val3])

    assert return_op.arg[0] is float_val1
    assert return_op.arg[1] is float_val2
    assert return_op.arg[2] is float_val3


def test_stencil_return_single_ResultType():
    result_type_val1 = TestSSAValue(ResultType(f32))
    return_op = ReturnOp.get([result_type_val1])

    assert return_op.arg[0] is result_type_val1


def test_stencil_return_multiple_ResultType():
    result_type_val1 = TestSSAValue(ResultType(f32))
    result_type_val2 = TestSSAValue(ResultType(f32))
    result_type_val3 = TestSSAValue(ResultType(f32))

    return_op = ReturnOp.get([result_type_val1, result_type_val2, result_type_val3])

    assert return_op.arg[0] is result_type_val1
    assert return_op.arg[1] is result_type_val2
    assert return_op.arg[2] is result_type_val3


def test_stencil_cast_op_verifier():
    typ = FieldType(3, f32)
    field = TestSSAValue(typ)

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
    non_dyn_field = TestSSAValue(FieldType(((-2, 102), (-2, 102), (-2, 102)), f32))
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
    field = TestSSAValue(FieldType(3, f32))

    cast = CastOp.get(
        field,
        StencilBoundsAttr(((-2, 100), (-3, 100), (-4, 0))),
    )

    assert cast.result.typ == FieldType(((-2, 100), (-3, 100), (-4, 0)), f32)


def test_stencil_apply():
    result_type_val1 = TestSSAValue(ResultType(f32))

    stencil_temptype = TempType(2, f32)
    apply_op = ApplyOp.get([result_type_val1], Block([]), [stencil_temptype])

    assert len(apply_op.args) == 1
    assert len(apply_op.res) == 1
    assert isinstance(apply_op.res[0].typ, TempType)
    assert len(apply_op.res[0].typ.get_shape()) == 2


def test_stencil_apply_no_args():
    stencil_temptype = TempType(1, f32)
    apply_op = ApplyOp.get([], Block([]), [stencil_temptype, stencil_temptype])

    assert len(apply_op.args) == 0
    assert len(apply_op.res) == 2
    assert isinstance(apply_op.res[0].typ, TempType)
    assert len(apply_op.res[0].typ.get_shape()) == 1


def test_stencil_apply_no_results():
    # Should error if there are no results expected
    with pytest.raises(AssertionError):
        ApplyOp.get([], Block([]), [])


@pytest.mark.parametrize(
    "indices",
    (
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
    ),
)
def test_create_index_attr_from_int_list(indices: list[int]):
    stencil_index_attr = IndexAttr.get(*indices)
    expected_array_attr = ArrayAttr(
        [(IntAttr(idx) if isinstance(idx, int) else idx) for idx in indices]
    )

    assert stencil_index_attr.array == expected_array_attr


def test_create_index_attr_from_list_edge_case1():
    with pytest.raises(VerifyException) as exc_info:
        IndexAttr.get()
    assert exc_info.value.args[0] == "Expected 1 to 3 indexes for stencil.index, got 0."


def test_create_index_attr_from_list_edge_case2():
    with pytest.raises(VerifyException) as exc_info:
        IndexAttr.get(*[1] * 4)
    assert exc_info.value.args[0] == "Expected 1 to 3 indexes for stencil.index, got 4."


@pytest.mark.parametrize(
    "indices1, indices2",
    (([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])),
)
def test_index_attr_size_from_bounds(indices1: list[int], indices2: list[int]):
    stencil_index_attr1 = IndexAttr.get(*indices1)
    stencil_index_attr2 = IndexAttr.get(*indices2)

    size_from_bounds = IndexAttr.size_from_bounds(
        stencil_index_attr1, stencil_index_attr2
    )
    expected_list = [abs(idx1 - idx2) for idx1, idx2 in zip(indices1, indices2)]

    assert size_from_bounds == expected_list


@pytest.mark.parametrize(
    "indices",
    (([1]), ([1, 2]), ([1, 2, 3])),
)
def test_index_attr_neg(indices: list[int]):
    stencil_index_attr = IndexAttr.get(*indices)
    stencil_index_attr_neg = -stencil_index_attr
    expected_array_attr = ArrayAttr([(IntAttr(-idx)) for idx in indices])

    assert stencil_index_attr_neg.array == expected_array_attr


@pytest.mark.parametrize(
    "indices1, indices2",
    (([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])),
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
    (([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])),
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
    (([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])),
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
    (([1], [4]), ([1, 2], [4, 5]), ([1, 2, 3], [5, 6, 7])),
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
    (((1,)), ((1, 2)), ((1, 2, 3))),
)
def test_index_attr_iter(indices: tuple[int]):
    stencil_index_attr = IndexAttr.get(*indices)

    assert tuple(stencil_index_attr) == indices


@pytest.mark.parametrize("indices", (([1]), ([1, 2]), ([1, 2, 3])))
def test_index_attr_indices_length(indices: list[int]):
    stencil_index_attr = IndexAttr.get(*indices)
    stencil_index_attr_iter = iter(stencil_index_attr)

    for idx in indices:
        assert idx == next(stencil_index_attr_iter)


@pytest.mark.parametrize(
    "attr, bounds",
    (
        (i32, ((0, 64), (0, 64))),
        (
            i64,
            ((0, 32), (0, 32), (0, 32)),
        ),
    ),
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
    (
        (i32, ((0, 1), (0, 2))),
        (i32, ((0, 1), (0, 1), (0, 3))),
        (i64, ((0, 1), (0, 1), (0, 3))),
    ),
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
    (
        (i32, []),
        (i64, []),
    ),
)
def test_stencil_fieldtype_constructor_empty_list(
    attr: IntegerType, bounds: list[tuple[int, int]]
):
    with pytest.raises(VerifyException) as exc_info:
        FieldType(bounds, attr)
    assert exc_info.value.args[0] == "Expected 1 to 3 indexes for stencil.index, got 0."


def test_stencil_load():
    field_type = FieldType([(0, 1), (0, 1)], f32)
    result_type_val1 = TestSSAValue(field_type)

    load = LoadOp.get(result_type_val1)

    assert isinstance(load.field.typ, FieldType)
    assert load.field.typ == field_type
    assert len(load.field.typ.get_shape()) == 2
    assert isinstance(load.field.typ.bounds, StencilBoundsAttr)
    assert isa(load.res.typ, TempType[Attribute])
    assert isa(load.res.typ.bounds, IntAttr)
    assert load.res.typ.bounds.data == 2


def test_stencil_load_bounds():
    field_type = FieldType([(0, 1), (0, 1)], f32)
    result_type_val1 = TestSSAValue(field_type)

    lb = IndexAttr.get(1, 1)
    ub = IndexAttr.get(64, 64)

    load = LoadOp.get(result_type_val1, lb, ub)

    assert isa(load.res.typ, TempType[Attribute])
    assert isinstance(load.res.typ.bounds, StencilBoundsAttr)
    assert isinstance(load.res.typ.bounds.lb, IndexAttr)
    assert isinstance(load.res.typ.bounds.ub, IndexAttr)
    assert len(load.res.typ.bounds.lb) == 2
    assert load.res.typ.bounds.lb == lb
    assert len(load.res.typ.bounds.ub) == 2
    assert load.res.typ.bounds.ub == ub


@pytest.mark.parametrize(
    "attr, dims",
    (
        (i32, ((0, 64), (0, 64))),
        (
            i64,
            ((0, 32), (0, 32), (0, 32)),
        ),
    ),
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
    (
        (i32, ((0, 1), (0, 2))),
        (i32, ((0, 1), (0, 1), (0, 3))),
        (i64, ((0, 1), (0, 1), (0, 3))),
    ),
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
    (
        (i32, []),
        (i64, []),
    ),
)
def test_stencil_temptype_constructor_empty_list(
    attr: IntegerType, dims: list[tuple[int, int]]
):
    with pytest.raises(VerifyException) as exc_info:
        TempType(dims, attr)
    assert exc_info.value.args[0] == "Expected 1 to 3 indexes for stencil.index, got 0."


@pytest.mark.parametrize(
    "float_type",
    ((bf16), (f16), (f32), (f64), (f80), (f128)),
)
def test_stencil_resulttype(float_type: AnyFloat):
    stencil_resulttype = ResultType(float_type)

    assert isinstance(stencil_resulttype, ResultType)
    assert stencil_resulttype.elem == float_type
