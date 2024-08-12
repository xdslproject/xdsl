from conftest import assert_print_op

from xdsl.dialects import test, transform
from xdsl.dialects.builtin import DenseArrayBase, IndexType, IntegerAttr, IntegerType
from xdsl.ir import Block, Region, SSAValue


def test_transform_op_type():
    opType = transform.OperationType("linalg.quantized_matmul")
    assert opType.name == "transform.op"
    assert opType.operation.data == "linalg.quantized_matmul"


def test_transform_param_type():
    paramType = transform.ParamType(transform.AnyValueType())
    assert paramType.name == "transform.param"
    assert paramType.type.name == "transform.any_value"


def test_transform_any_op_type():
    anyOpType = transform.AnyOpType()
    assert anyOpType.name == "transform.any_op"


def test_transform_any_value_type():
    anyValueType = transform.AnyValueType()
    assert anyValueType.name == "transform.any_value"


def test_transform_affine_map_type():
    affineMapType = transform.AffineMapType()
    assert affineMapType.name == "transform.affine_map"


def test_sequence_init():
    failurePropagationMode = IntegerAttr(1, IntegerType(32))
    root: list[SSAValue] = []
    extra_bindings: list[SSAValue] = []
    region = Region(
        Block(
            arg_types=[
                transform.AnyValueType(),
                transform.OperationType("linalg.matmul"),
            ]
        ),
    )
    seq = transform.SequenceOp(
        body=region,
        failure_propagation_mode=failurePropagationMode,
        root=root,
        extra_bindings=extra_bindings,
    )

    expected = """
"transform.sequence"() <{"failure_propagation_mode" = 1 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> ({
^0(%0 : !transform.any_value, %1 : !transform.op<"linalg.matmul">):
}) : () -> ()
"""
    assert_print_op(seq, expected, None)


def test_tileop_init():
    block = Block(
        arg_types=[
            transform.AnyValueType(),
            transform.OperationType("linalg.matmul"),
        ]
    )

    target = block.args[0]
    static_sizes = DenseArrayBase.create_dense_int_or_index(IndexType(), [8, 8])
    tile_op = transform.TileOp(
        target=target,
        dynamic_sizes=[],
        static_sizes=static_sizes,
    )

    expected = """
%0, %1, %2 = "transform.structured.tile_using_for"(%3) <{"static_sizes" = array<index: 8, 8>}> : (!transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op)"""
    assert_print_op(tile_op, expected, None)
    assert isinstance(tile_op.results[0], SSAValue)


def test_get_consumer_of_result():
    result_number = 0
    target = test.TestOp(result_types=[transform.AnyOpType()]).results[0]
    op = transform.GetConsumerOfResult(target=target, result_number=result_number)
    expected = """
    %0 = "transform.get_consumers_of_result"(%1) <{"result_number" = 0 : i64}> : (!transform.any_op) -> !transform.any_op
    """
    assert_print_op(op, expected, None)
    assert isinstance(op.results[0], SSAValue)


def test_defining_op():
    target = test.TestOp(result_types=[transform.AnyOpType()]).results[0]
    op = transform.GetDefiningOp(target=target)
    expected = """
    %0 = "transform.get_defining_op"(%1) : (!transform.any_op) -> !transform.any_op    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


def test_get_parent_op():
    target = test.TestOp(result_types=[transform.AnyOpType()]).results[0]
    op = transform.GetParentOp(target=target)
    expected = """
%0 = "transform.get_parent_op"(%1) <{"nth_parent" = 1 : i64}> : (!transform.any_op) -> !transform.any_op    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


def test_get_producer_of_operand():
    target = test.TestOp(result_types=[transform.AnyValueType()]).results[0]
    op = transform.GetProducerOfOperand(operand_number=0, target=target)
    expected = """
%0 = "transform.get_producer_of_operand"(%1) <{"operand_number" = 0 : i64}> : (!transform.any_value) -> !transform.any_op
"""
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


def test_get_result():
    target = test.TestOp(result_types=[transform.AnyOpType()]).results[0]
    result_number = 0
    op = transform.GetResultOp(target=target, result_number=result_number)
    expected = """
%0 = "transform.get_result"(%1) <{"result_number" = 0 : i64}> : (!transform.any_op) -> !transform.any_value
"""
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


def test_get_type():
    value = test.TestOp(result_types=[transform.AnyValueType()]).results[0]
    op = transform.GetTypeOp(elemental=False, value=value)
    expected = """
    %0 = "transform.get_type"(%1) : (!transform.any_value) -> !transform.any_param
"""
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


def test_include():
    target = "foo"
    operands_input = [test.TestOp(result_types=[transform.AnyValueType()]).results[0]]
    op = transform.IncludeOp(
        target=target, failure_propagation_mode=0, operands_input=operands_input
    )
    expected = """
    %0 = "transform.include"(%1) <{"target" = @foo, "failure_propagation_mode" = false}> : (!transform.any_value) -> !transform.any_value
    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


def test_match_empty():
    handle = test.TestOp(result_types=[transform.AnyOpType()]).results[0]
    op = transform.MatchOperationEmptyOp(operand_handle=handle)
    expected = """
    "transform.match.operation_empty"(%0) : (!transform.any_op) -> ()
    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert len(op.results) == 0


def test_match_name():
    handle = test.TestOp(result_types=[transform.AnyOpType()]).results[0]
    op = transform.MatchOperationNameOp(operand_handle=handle, op_names=["foo"])
    expected = """
    "transform.match.operation_name"(%0) <{"op_names" = ["foo"]}> : (!transform.any_op) -> ()    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert len(op.results) == 0


def test_match_param():
    predicate = 0
    param = test.TestOp(result_types=[transform.AnyParamType()]).results[0]
    reference = test.TestOp(result_types=[transform.AnyParamType()]).results[0]
    op = transform.MatchParamCmpIOp(
        predicate=predicate, param=param, reference=reference
    )
    expected = """
    "transform.match.param.cmpi"(%0, %1) <{"predicate" = 0 : i64}> : (!transform.any_param, !transform.any_param) -> ()
    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert len(op.results) == 0


def test_merge_handles():
    handles = [test.TestOp(result_types=[transform.AnyOpType()]).results[0]]
    op = transform.MergeHandlesOp(handles=handles, deduplicate=True)
    expected = """
    %0 = "transform.merge_handles"(%1) <{"deduplicate"}> : (!transform.any_op) -> !transform.any_op    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


def test_param_const():
    value = IntegerAttr(1, IntegerType(32))
    op = transform.ParamConstantOp(value=value, param_type=IntegerType(32))
    expected = """
    %0 = "transform.param.constant"() <{"value" = 1 : i32}> : () -> !transform.param<i32>    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


def test_split_handle():
    handle = test.TestOp(result_types=[transform.AnyOpType()]).results[0]
    op = transform.SplitHandleOp(
        handle=handle,
        number_of_results=2,
        pass_through_empty_handle=True,
        fail_on_payload_too_small=True,
        overflow_result=1,
    )
    expected = """
    %0, %1 = "transform.split_handle"(%2) <{"pass_through_empty_handle" = true, "fail_on_payload_too_small" = true, "overflow_result" = 1 : i64}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    """
    assert_print_op(
        op,
        expected,
        None,
    )
    assert isinstance(op.results[0], SSAValue)


test_tileop_init()
