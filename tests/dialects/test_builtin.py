from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, f32, FloatAttr


def test_DenseIntOrFPElementsAttr_fp_type_conversion():
    check1 = DenseIntOrFPElementsAttr.tensor_from_list([4, 5], f32)

    value1 = check1.data.data[0].value.data
    value2 = check1.data.data[1].value.data

    # Ensure type conversion happened properly during attribute construction.
    assert type(value1) == float
    assert value1 == 4.0
    assert type(value2) == float
    assert value2 == 5.0

    t1 = FloatAttr.from_value(4.0, f32)
    t2 = FloatAttr.from_value(5.0, f32)

    check2 = DenseIntOrFPElementsAttr.tensor_from_list([t1, t2], f32)

    value3 = check2.data.data[0].value.data
    value4 = check2.data.data[1].value.data

    # Ensure type conversion happened properly during attribute construction.
    assert type(value3) == float
    assert value3 == 4.0
    assert type(value4) == float
    assert value4 == 5.0


# TODO: Add more tests for the builtin dialect.
