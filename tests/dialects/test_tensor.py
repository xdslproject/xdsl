from xdsl.dialects.builtin import DYNAMIC_INDEX, DenseArrayBase, TensorType, f64, i64
from xdsl.dialects.stencil import IndexAttr
from xdsl.dialects.tensor import ExtractSliceOp, FromElementsOp, InsertSliceOp
from xdsl.dialects.test import TestOp
from xdsl.utils.test_value import create_ssa_value


def test_extract_slice_static():
    input_t = TensorType(f64, [10, 20, 30])
    input_v = TestOp(result_types=[input_t]).res[0]

    extract_slice = ExtractSliceOp.from_static_parameters(input_v, [1, 2, 3], [4, 5, 6])

    assert extract_slice.source is input_v
    assert extract_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2, 3])
    assert extract_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5, 6])
    assert extract_slice.static_strides == DenseArrayBase.from_list(i64, [1, 1, 1])
    assert extract_slice.offsets == ()
    assert extract_slice.sizes == ()
    assert extract_slice.strides == ()
    assert extract_slice.result.type == TensorType(f64, [4, 5, 6])

    extract_slice = ExtractSliceOp.from_static_parameters(
        input_v, [1, 2, 3], [4, 5, 6], [8, 9, 10]
    )

    assert extract_slice.source is input_v
    assert extract_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2, 3])
    assert extract_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5, 6])
    assert extract_slice.static_strides == DenseArrayBase.from_list(i64, [8, 9, 10])
    assert extract_slice.offsets == ()
    assert extract_slice.sizes == ()
    assert extract_slice.strides == ()
    assert extract_slice.result.type == TensorType(f64, [4, 5, 6])


def test_insert_slice_static():
    source_t = TensorType(f64, [10, 20])
    source_v = TestOp(result_types=[source_t]).res[0]
    dest_t = TensorType(f64, [10, 20, 30])
    dest_v = TestOp(result_types=[dest_t]).res[0]

    insert_slice = InsertSliceOp.from_static_parameters(
        source_v, dest_v, [1, 2], [4, 5]
    )

    assert insert_slice.source is source_v
    assert insert_slice.dest is dest_v
    assert insert_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2])
    assert insert_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5])
    assert insert_slice.static_strides == DenseArrayBase.from_list(i64, [1, 1])
    assert insert_slice.offsets == ()
    assert insert_slice.sizes == ()
    assert insert_slice.strides == ()
    assert insert_slice.result.type == dest_t

    insert_slice = InsertSliceOp.from_static_parameters(
        source_v, dest_v, [1, 2], [4, 5], [8, 9]
    )

    assert insert_slice.source is source_v
    assert insert_slice.dest is dest_v
    assert insert_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2])
    assert insert_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5])
    assert insert_slice.static_strides == DenseArrayBase.from_list(i64, [8, 9])
    assert insert_slice.offsets == ()
    assert insert_slice.sizes == ()
    assert insert_slice.strides == ()
    assert insert_slice.result.type == dest_t


def test_insert_slice_dynamic():
    source_t = TensorType(f64, [10, 20])
    source_v = create_ssa_value(source_t)
    dest_t = TensorType(f64, [10, 20, 30])
    dest_v = create_ssa_value(dest_t)
    offset1 = create_ssa_value(IndexAttr.get(3))
    offset2 = create_ssa_value(IndexAttr.get(15))
    stride1 = create_ssa_value(IndexAttr.get(2))
    stride2 = create_ssa_value(IndexAttr.get(5))

    insert_slice = InsertSliceOp.get(
        source=source_v,
        dest=dest_v,
        static_sizes=[1, 2],
        offsets=[offset1, offset2],
        strides=[stride1, stride2],
    )

    assert insert_slice.static_offsets == DenseArrayBase.from_list(
        i64, 2 * [DYNAMIC_INDEX]
    )
    assert insert_slice.static_strides == DenseArrayBase.from_list(
        i64, 2 * [DYNAMIC_INDEX]
    )


def test_from_elements_scalar():
    """Test FromElementsOp with a single scalar element (0-D tensor)."""
    a = create_ssa_value(i64)

    res = FromElementsOp(a, result_type=TensorType(i64, shape=tuple()))

    # Check that the result is a 0-D tensor (scalar)
    assert isinstance(res.result.type, TensorType)
    assert res.result.type.get_shape() == ()
    assert res.result.type.element_type == i64
    assert len(res.elements) == 1
    assert res.elements[0] is a


def test_from_elements_1d_tensor():
    """Test FromElementsOp with multiple elements (1-D tensor)."""
    a = create_ssa_value(i64)
    b = create_ssa_value(i64)
    c = create_ssa_value(i64)

    res = FromElementsOp(a, b, c)

    # Check that the result is a 1-D tensor with 3 elements
    assert isinstance(res.result.type, TensorType)
    assert res.result.type.get_shape() == (3,)
    assert res.result.type.element_type == i64
    assert len(res.elements) == 3
    assert res.elements[0] is a
    assert res.elements[1] is b
    assert res.elements[2] is c


def test_from_elements_single_element():
    """Test FromElementsOp with a single element in a list."""
    a = create_ssa_value(f64)

    res = FromElementsOp(a)

    # Check that the result is a 1-D tensor with 1 element
    assert isinstance(res.result.type, TensorType)
    assert res.result.type.get_shape() == (1,)
    assert res.result.type.element_type == f64
    assert len(res.elements) == 1
    assert res.elements[0] is a


def test_from_elements_empty_list():
    """Test FromElementsOp with an empty list."""
    # Empty lists should raise a ValueError since we can't infer element type
    with pytest.raises(ValueError):
        FromElementsOp()


def test_from_elements_different_numeric_types():
    """Test FromElementsOp with different numeric element types."""
    # Test with f64
    a_f64 = create_ssa_value(f64)
    b_f64 = create_ssa_value(f64)

    res_f64 = FromElementsOp(a_f64, b_f64)
    assert res_f64.result.type.element_type == f64
    assert res_f64.result.type.get_shape() == (2,)

    # Test with i64
    a_i64 = create_ssa_value(i64)
    b_i64 = create_ssa_value(i64)

    res_i64 = FromElementsOp(a_i64, b_i64)
    assert res_i64.result.type.element_type == i64
    assert res_i64.result.type.get_shape() == (2,)


def test_from_elements_type_consistency():
    """Test that FromElementsOp enforces type consistency among elements."""
    a_i64 = create_ssa_value(i64)
    b_f64 = create_ssa_value(f64)

    from xdsl.utils.exceptions import VerifyException

    # This should raise an assertion error due to type mismatch
    try:
        FromElementsOp(a_i64, b_f64).verify()
    except VerifyException:
        # This is expected
        return
    raise Exception("Expected assertion error for mismatched types")


def test_from_elements_large_tensor():
    """Test FromElementsOp with a larger number of elements."""
    elements = [create_ssa_value(i64) for _ in range(10)]

    res = FromElementsOp(*elements)

    assert isinstance(res.result.type, TensorType)
    assert res.result.type.get_shape() == (100,)
    assert res.result.type.element_type == i64
    assert len(res.elements) == 100


def test_from_elements_assembly_format():
    """Test FromElementsOp assembly format parsing and printing."""
    from io import StringIO

    from xdsl.printer import Printer

    # Test printing
    a = create_ssa_value(i64)
    b = create_ssa_value(i64)

    res = FromElementsOp(a, b)

    output = StringIO()
    printer = Printer(stream=output)
    res.print(printer)

    # The assembly format should be: elements attr-dict : type(result)
    assembly = output.getvalue()
    # The print method prints only the assembly format part, not the operation name
    assert "tensor<2xi64>" in assembly
    assert "%0, %1" in assembly  # The operands should be printed


def test_from_elements_verify_trait():
    """Test that FromElementsOp has NoMemoryEffect trait."""
    a = create_ssa_value(i64)
    res = FromElementsOp(a)

    # Check that the operation has the NoMemoryEffect trait
    from xdsl.traits import NoMemoryEffect

    assert any(isinstance(trait, NoMemoryEffect) for trait in res.traits)
