# RUN: python %s | filecheck %s

from collections.abc import Callable

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import builtin, func, get_all_dialects, linalg, transform
from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
from xdsl.dialects.linalg.transforms.tiling import tile_structured_op
from xdsl.ir import Block, Operation, Region
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.transforms.transform_interpreter import TransformInterpreterPass


def apply_func(op: func.FuncOp, func: Callable[[Operation], None]) -> func.FuncOp:
    clone = op.clone()
    for _op in tuple(clone.walk(region_first=True)):
        func(_op)
    clone.verify()
    return clone


def apply_transform(
    op: func.FuncOp, sequence: transform.NamedSequenceOp
) -> func.FuncOp:
    ctx = Context()
    for name, factory in get_all_dialects().items():
        ctx.register_dialect(name, factory)

    assert sequence.parent is None
    schedule_module = builtin.ModuleOp(
        [transform_result := op.clone(), sequence],
        attributes={"transform.with_named_sequence": builtin.UnitAttr()},
    )
    schedule_module.verify()

    # Also compare with the textual schedule in tile.mlir, so both Python paths
    # must agree with the existing CLI regression, not just with each other.
    TransformInterpreterPass().apply(ctx, schedule_module)
    schedule_module.verify()

    return transform_result.clone()


def tile(op: Operation) -> None:
    """Tile a structured linalg operation by 32 in each of its three dimensions."""
    if not isinstance(op, LinalgStructuredOperation):
        return
    rewriter = PatternRewriter(op)
    tile_structured_op(rewriter, op, (32, 32, 32))


def build_named_sequence() -> transform.NamedSequenceOp:
    """Build the equivalent matmul schedule without creating its outer module."""
    block = Block(arg_types=[transform.AnyOpType()])
    with ImplicitBuilder(block):
        matmul = transform.MatchOp(block.args[0], ops=["linalg.matmul"])
        transform.TileOp(matmul.result, static_sizes=[32, 32, 32])
        transform.YieldOp()
    return transform.NamedSequenceOp(
        "__transform_main",
        ([transform.AnyOpType()], []),
        Region(block),
        arg_attrs=[builtin.DictionaryAttr({"transform.readonly": builtin.UnitAttr()})],
    )


#   func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
#     %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
#     return %0 : tensor<128x128xf32>
#   }
def build_input() -> func.FuncOp:
    tensor_type = builtin.TensorType(builtin.f32, (128, 128))
    func_op = func.FuncOp(
        "matmul", ((tensor_type, tensor_type, tensor_type), (tensor_type,))
    )
    with ImplicitBuilder(func_op.body) as (a, b, c):
        a.name_hint = "A"
        b.name_hint = "B"
        c.name_hint = "C"
        matmul_op = linalg.ops.MatmulOp((a, b), (c,))
        func.ReturnOp(*matmul_op.results)

    return func_op


python_result = apply_func(build_input(), tile)
transform_result = apply_transform(build_input(), build_named_sequence())

assert python_result.is_structurally_equivalent(transform_result), (
    "The constructed named sequence and the fixture produced different IR"
)

print(python_result)

#      CHECK:  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
# CHECK-NEXT:    %0 = arith.constant 0 : index
# CHECK-NEXT:    %1 = arith.constant 128 : index
# CHECK-NEXT:    %2 = arith.constant 128 : index
# CHECK-NEXT:    %3 = arith.constant 128 : index
# CHECK-NEXT:    %4 = arith.constant 32 : index
# CHECK-NEXT:    %5 = arith.constant 32 : index
# CHECK-NEXT:    %6 = arith.constant 32 : index
# CHECK-NEXT:    %7 = scf.for %8 = %0 to %1 step %4 iter_args(%9 = %C) -> (tensor<128x128xf32>) {
# CHECK-NEXT:      %10 = scf.for %11 = %0 to %2 step %5 iter_args(%12 = %9) -> (tensor<128x128xf32>) {
# CHECK-NEXT:        %13 = scf.for %14 = %0 to %3 step %6 iter_args(%15 = %12) -> (tensor<128x128xf32>) {
# CHECK-NEXT:          %16 = tensor.extract_slice %A[%8, %14] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
# CHECK-NEXT:          %17 = tensor.extract_slice %B[%14, %11] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
# CHECK-NEXT:          %18 = tensor.extract_slice %15[%8, %11] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
# CHECK-NEXT:          %19 = linalg.matmul ins(%16, %17 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
# CHECK-NEXT:          %20 = tensor.insert_slice %19 into %15[%8, %11] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
# CHECK-NEXT:          scf.yield %20 : tensor<128x128xf32>
# CHECK-NEXT:        }
# CHECK-NEXT:        scf.yield %13 : tensor<128x128xf32>
# CHECK-NEXT:      }
# CHECK-NEXT:      scf.yield %10 : tensor<128x128xf32>
# CHECK-NEXT:    }
# CHECK-NEXT:    func.return %7 : tensor<128x128xf32>
# CHECK-NEXT:  }
