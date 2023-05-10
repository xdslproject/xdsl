import pytest

from xdsl.dialects.builtin import (
    FloatAttr,
    IntegerAttr,
    f32,
    f64,
    i32,
    i64,
    IntegerType,
    ArrayAttr,
    AnyIntegerAttr,
)
from xdsl.dialects.experimental.stencil import (
    ReturnOp,
    ResultType,
    ApplyOp,
    TempType,
    LoadOp,
    FieldType,
    IndexAttr,
)
from xdsl.dialects.stencil import CastOp
from xdsl.ir import Block
from xdsl.utils.exceptions import VerifyException
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
    result_type_val1 = TestSSAValue(ResultType.from_type(f32))
    return_op = ReturnOp.get([result_type_val1])

    assert return_op.arg[0] is result_type_val1


def test_stencil_return_multiple_ResultType():
    result_type_val1 = TestSSAValue(ResultType.from_type(f32))
    result_type_val2 = TestSSAValue(ResultType.from_type(f32))
    result_type_val3 = TestSSAValue(ResultType.from_type(f32))

    return_op = ReturnOp.get([result_type_val1, result_type_val2, result_type_val3])

    assert return_op.arg[0] is result_type_val1
    assert return_op.arg[1] is result_type_val2
    assert return_op.arg[2] is result_type_val3


def test_stencil_cast_op_verifier():
    field = TestSSAValue(FieldType((-1, -1, -1), f32))

    # check that correct op verifies correctly
    cast = CastOp.get(
        field,
        IndexAttr.get(-2, -2, -2),
        IndexAttr.get(100, 100, 100),
        FieldType((102, 102, 102), f32),
    )
    cast.verify()

    # check that math is correct
    with pytest.raises(VerifyException, match="math"):
        cast = CastOp.get(
            field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType((100, 100, 100), f32),
        )
        cast.verify()

    # check that output has same dims as input and lb, ub
    with pytest.raises(VerifyException, match="same dimensions"):
        cast = CastOp.get(
            field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType((102, 102), f32),
        )
        cast.verify()

    # check that input has same shape as lb, ub, output
    with pytest.raises(VerifyException, match="same dimensions"):
        dyn_field_wrong_shape = TestSSAValue(FieldType((-1, -1), f32))
        cast = CastOp.get(
            dyn_field_wrong_shape,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType((102, 102, 102), f32),
        )
        cast.verify()

    # check that input and output have same element type
    with pytest.raises(VerifyException, match="element type"):
        cast = CastOp.get(
            field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 100),
            FieldType((102, 102, 102), f64),
        )
        cast.verify()

    # check that len(lb) == len(ub)
    with pytest.raises(VerifyException, match="same dimensions"):
        cast = CastOp.get(
            field,
            IndexAttr.get(
                -2,
                -2,
            ),
            IndexAttr.get(100, 100, 100),
            FieldType((102, 102, 102), f32),
        )
        cast.verify()

    # check that len(lb) == len(ub)
    with pytest.raises(VerifyException, match="same dimensions"):
        cast = CastOp.get(
            field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100),
            FieldType((102, 102, 102), f32),
        )
        cast.verify()

    # check that non-dynamic input verifies
    non_dyn_field = TestSSAValue(FieldType((102, 102, 102), f32))
    cast = CastOp.get(
        non_dyn_field,
        IndexAttr.get(-2, -2, -2),
        IndexAttr.get(100, 100, 100),
        FieldType((102, 102, 102), f32),
    )
    cast.verify()

    with pytest.raises(
        VerifyException,
        match="If input shape is not dynamic, it must be the same as output",
    ):
        cast = CastOp.get(
            non_dyn_field,
            IndexAttr.get(-2, -2, -2),
            IndexAttr.get(100, 100, 101),
            FieldType((102, 102, 103), f32),
        )
        cast.verify()


def test_cast_op_constructor():
    field = TestSSAValue(FieldType((-1, -1, -1), f32))

    cast = CastOp.get(
        field,
        IndexAttr.get(-2, -3, -4),
        IndexAttr.get(100, 100, 0),
    )

    assert cast.result.typ == FieldType((102, 103, 4), f32)


def test_stencil_apply():
    result_type_val1 = TestSSAValue(ResultType.from_type(f32))

    stencil_temptype = TempType([-1] * 2, f32)
    apply_op = ApplyOp.get([result_type_val1], Block([]), [stencil_temptype])

    assert len(apply_op.args) == 1
    assert len(apply_op.res) == 1
    assert isinstance(apply_op.res[0].typ, TempType)
    assert len(apply_op.res[0].typ.shape) == 2


def test_stencil_apply_no_args():
    stencil_temptype = TempType([-1] * 1, f32)
    apply_op = ApplyOp.get([], Block([]), [stencil_temptype, stencil_temptype])

    assert len(apply_op.args) == 0
    assert len(apply_op.res) == 2
    assert isinstance(apply_op.res[0].typ, TempType)
    assert len(apply_op.res[0].typ.shape) == 1


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
                IntegerAttr[IntegerType](1, 64),
                IntegerAttr[IntegerType](2, 64),
                IntegerAttr[IntegerType](3, 64),
            ]
        ),
    ),
)
def test_create_index_attr_from_int_list(indices: list[int]):
    stencil_index_attr = IndexAttr.get(*indices)
    expected_array_attr = ArrayAttr(
        [
            (IntegerAttr[IntegerType](idx, 64) if isinstance(idx, int) else idx)
            for idx in indices
        ]
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
    expected_array_attr = ArrayAttr(
        [(IntegerAttr[IntegerType](-idx, 64)) for idx in indices]
    )

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
        [
            (IntegerAttr[IntegerType](idx1 + idx2, 64))
            for idx1, idx2 in zip(indices1, indices2)
        ]
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
        [
            (IntegerAttr[IntegerType](idx1 - idx2, 64))
            for idx1, idx2 in zip(indices1, indices2)
        ]
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
        [
            (IntegerAttr[IntegerType](min(idx1, idx2), 64))
            for idx1, idx2 in zip(indices1, indices2)
        ]
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
        [
            (IntegerAttr[IntegerType](max(idx1, idx2), 64))
            for idx1, idx2 in zip(indices1, indices2)
        ]
    )

    assert stencil_index_attr_max.array == expected_array_attr


@pytest.mark.parametrize(
    "indices",
    (([1]), ([1, 2]), ([1, 2, 3])),
)
def test_index_attr_tuple_return(indices: list[int]):
    stencil_index_attr = IndexAttr.get(*indices)

    assert stencil_index_attr.as_tuple() == tuple(indices)


@pytest.mark.parametrize(
    "indices",
    (([1]), ([1, 2]), ([1, 2, 3])),
)
def test_index_attr_indices_length(indices: list[int]):
    stencil_index_attr = IndexAttr.get(*indices)
    stencil_index_attr_iter = iter(stencil_index_attr)

    for idx in indices:
        assert idx == next(stencil_index_attr_iter)


@pytest.mark.parametrize(
    "attr, dims",
    (
        (
            i32,
            ArrayAttr(
                [IntegerAttr[IntegerType](1, 64), IntegerAttr[IntegerType](2, 64)]
            ),
        ),
        (
            i64,
            ArrayAttr(
                [
                    IntegerAttr[IntegerType](1, 32),
                    IntegerAttr[IntegerType](2, 32),
                    IntegerAttr[IntegerType](3, 32),
                ]
            ),
        ),
    ),
)
def test_stencil_fieldtype_constructor_with_ArrayAttr(
    attr: IntegerType, dims: ArrayAttr[AnyIntegerAttr]
):
    stencil_fieldtype = FieldType(dims, attr)

    assert stencil_fieldtype.element_type == attr
    assert stencil_fieldtype.get_num_dims() == len(dims)
    assert stencil_fieldtype.get_shape() == [
        list(dims.data)[dim].value.data for dim in range(len(dims))
    ]


@pytest.mark.parametrize(
    "attr, dims",
    (
        (i32, [1, 2]),
        (i32, [1, 1, 3]),
        (i64, [1, 1, 3]),
    ),
)
def test_stencil_fieldtype_constructor(attr: IntegerType, dims: list[int]):
    stencil_fieldtype = FieldType(dims, attr)

    assert stencil_fieldtype.element_type == attr
    assert stencil_fieldtype.get_num_dims() == len(dims)
    assert stencil_fieldtype.get_shape() == dims


@pytest.mark.parametrize(
    "attr, dims",
    (
        (i32, []),
        (i64, []),
    ),
)
def test_stencil_fieldtype_constructor_empty_list(attr: IntegerType, dims: list[int]):
    with pytest.raises(VerifyException) as exc_info:
        FieldType(dims, attr)
    assert (
        exc_info.value.args[0]
        == "Number of field dimensions must be greater than zero, got 0."
    )


def test_stencil_load():
    field_type = FieldType([1, 1], f32)
    result_type_val1 = TestSSAValue(field_type)

    load = LoadOp.get(result_type_val1)

    assert isinstance(load.field.typ, FieldType)
    assert load.field.typ == field_type
    assert len(load.field.typ.shape) == 2
    assert load.lb is None
    assert load.ub is None


def test_stencil_load_bounds():
    field_type = FieldType([1, 1], f32)
    result_type_val1 = TestSSAValue(field_type)

    lb = IndexAttr.get(1, 1)
    ub = IndexAttr.get(64, 64)

    load = LoadOp.get(result_type_val1, lb, ub)

    assert isinstance(load.lb, IndexAttr)
    assert len(load.lb.array) == 2
    for my_val, load_val in zip(lb.array.data, load.lb.array):
        assert my_val.value.data == load_val.value.data
    assert isinstance(load.ub, IndexAttr)
    assert len(load.ub.array) == 2
    for my_val, load_val in zip(ub.array.data, load.ub.array):
        assert my_val.value.data == load_val.value.data


@pytest.mark.parametrize(
    "attr, dims",
    (
        (
            i32,
            ArrayAttr(
                [IntegerAttr[IntegerType](1, 64), IntegerAttr[IntegerType](2, 64)]
            ),
        ),
        (
            i64,
            ArrayAttr(
                [
                    IntegerAttr[IntegerType](1, 32),
                    IntegerAttr[IntegerType](2, 32),
                    IntegerAttr[IntegerType](3, 32),
                ]
            ),
        ),
    ),
)
def test_stencil_temptype_constructor_with_ArrayAttr(
    attr: IntegerType, dims: ArrayAttr[AnyIntegerAttr]
):
    stencil_temptype = TempType(dims, attr)

    assert isinstance(stencil_temptype, TempType)
    assert stencil_temptype.element_type == attr
    assert stencil_temptype.get_num_dims() == len(dims)
    assert stencil_temptype.get_shape() == [
        list(dims.data)[dim].value.data for dim in range(len(dims))
    ]


@pytest.mark.parametrize(
    "attr, dims",
    (
        (i32, [1, 2]),
        (i32, [1, 1, 3]),
        (i64, [1, 1, 3]),
    ),
)
def test_stencil_temptype_constructor(attr: IntegerType, dims: list[int]):
    stencil_temptype = TempType(dims, attr)

    assert isinstance(stencil_temptype, TempType)
    assert stencil_temptype.element_type == attr
    assert stencil_temptype.get_num_dims() == len(dims)
    assert stencil_temptype.get_shape() == dims


@pytest.mark.parametrize(
    "attr, dims",
    (
        (i32, []),
        (i64, []),
    ),
)
def test_stencil_temptype_constructor_empty_list(attr: IntegerType, dims: list[int]):
    with pytest.raises(VerifyException) as exc_info:
        TempType(dims, attr)
    assert (
        exc_info.value.args[0]
        == "Number of field dimensions must be greater than zero, got 0."
    )


@pytest.mark.parametrize(
    "attr, dims",
    (
        (i32, [1, 2]),
        (i32, [1, 1, 3]),
        (i64, [1, 1, 3]),
    ),
)
def test_stencil_temptype_printing(attr: IntegerType, dims: list[int]):
    stencil_temptype = TempType(dims, attr)

    expected_string: str = "stencil.Temp<["
    for dim in dims:
        expected_string += f"{dim} "
    expected_string += "]>"

    assert repr(stencil_temptype) == expected_string
