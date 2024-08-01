from conftest import assert_print_op

from xdsl.dialects import transform
from xdsl.dialects.builtin import DenseArrayBase, IndexType, IntegerAttr, IntegerType
from xdsl.ir import Block, Region, SSAValue
from xdsl.printer import Printer


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


def test_transform_any_param_type():
    anyParamType = transform.AnyParamType()
    assert anyParamType.name == "transform.any_param"


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

    printer = Printer()
    printer.print(tile_op)
    expected = """
%0, %1 = "transform.structured.tile"(%2) <{"static_sizes" = array<index: 8, 8>}> : (!transform.any_value) -> (!transform.any_op, !transform.any_op)
"""
    assert_print_op(tile_op, expected, None)
    assert isinstance(tile_op.results[0], SSAValue)
