# RUN: python %s | filecheck %s

import ast
from collections.abc import Callable
from typing import Any

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, get_all_dialects, linalg, scf, transform
from xdsl.dialects.linalg.transforms.tiling import tile_structured_op
from xdsl.frontend.pyast.code_generation import CodeGenerationVisitor
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.program import PyASTProgram
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.interpreters.transform import OperationHandle
from xdsl.ir import Dialect, Operation, OperationInvT, OpResult, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import Rewriter
from xdsl.traits import Pure
from xdsl.transforms.dead_code_elimination import region_dce
from xdsl.transforms.desymref import FrontendDesymrefyPass
from xdsl.transforms.transform_interpreter import TransformInterpreterPass
from xdsl.utils.exceptions import DiagnosticException

SIZES_TYPE = builtin.TupleType((builtin.i32, builtin.i32, builtin.i32))


@irdl_op_definition
class SizesOp(IRDLOperation):
    """Pack three integer SSA values, preserving the Python tuple until lowering."""

    name = "pytransform.sizes"

    i = operand_def(builtin.i32)
    j = operand_def(builtin.i32)
    k = operand_def(builtin.i32)
    result = result_def(SIZES_TYPE)

    traits = traits_def(Pure())

    def __init__(self, i: SSAValue, j: SSAValue, k: SSAValue):
        super().__init__(operands=[i, j, k], result_types=[SIZES_TYPE])


@irdl_op_definition
class TileOp(IRDLOperation):
    """
    Tile a payload using sizes that will be resolved to properties by a pass.

    The prototype supports three positive constant sizes and returns four handles.
    This operation represents a payload mutation, even when all results are unused.
    """

    name = "pytransform.tile"

    target = operand_def(transform.AnyOpType)
    sizes = operand_def(SIZES_TYPE)
    tiled = result_def(transform.AnyOpType)
    i = result_def(transform.AnyOpType)
    j = result_def(transform.AnyOpType)
    k = result_def(transform.AnyOpType)

    def __init__(self, target: SSAValue, sizes: SSAValue):
        super().__init__(
            operands=[target, sizes], result_types=[transform.AnyOpType()] * 4
        )


PyTransform = Dialect("pytransform", [SizesOp, TileOp], [])


class LowerPyTransformPass(ModulePass):
    """Lower one straight-line schedule with three positive constant tile sizes."""

    name = "lower-pytransform"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        functions = tuple(op.ops)
        if len(functions) != 1 or not isinstance(function := functions[0], func.FuncOp):
            raise DiagnosticException("expected exactly one Python schedule function")
        if (
            function.function_type.inputs.data != (transform.AnyOpType(),)
            or function.function_type.outputs.data
            or len(function.body.blocks) != 1
        ):
            raise DiagnosticException(
                "expected a single-block schedule with one !transform.any_op argument "
                "and no results"
            )
        block = function.body.block
        terminator = block.last_op
        if not isinstance(terminator, func.ReturnOp) or terminator.arguments:
            raise DiagnosticException("expected a schedule ending in a bare return")

        for helper in tuple(block.ops):
            if not isinstance(helper, TileOp):
                continue
            if not isinstance(helper.sizes, OpResult) or not isinstance(
                sizes_op := helper.sizes.owner, SizesOp
            ):
                raise DiagnosticException("expected tile sizes from pytransform.sizes")
            sizes: list[int] = []
            for operand in sizes_op.operands:
                if (
                    not isinstance(operand, OpResult)
                    or not isinstance(constant := operand.owner, arith.ConstantOp)
                    or not isinstance(constant.value, builtin.IntegerAttr)
                ):
                    raise DiagnosticException(
                        "expected tile sizes from arith.constant i32 operations"
                    )
                size = constant.value.value.data
                if size <= 0:
                    raise DiagnosticException(
                        "the Python transform prototype requires positive tile sizes"
                    )
                sizes.append(size)
            Rewriter.replace_op(
                helper,
                transform.TileOp(
                    helper.target, static_sizes=sizes, scalable_sizes=[0] * 3
                ),
            )

        # Delete the dead tuple and constants, but retain payload transformations.
        region_dce(function.body)
        if any(
            not isinstance(
                body_op, transform.MatchOp | transform.TileOp | func.ReturnOp
            )
            for body_op in block.ops
        ):
            raise DiagnosticException(
                "unsupported operation in Python transform schedule"
            )

        Rewriter.replace_op(terminator, transform.YieldOp())
        sequence = transform.NamedSequenceOp(
            "__transform_main",
            function.function_type,
            function.detach_region(function.body),
            arg_attrs=[
                builtin.DictionaryAttr({"transform.readonly": builtin.UnitAttr()})
            ],
        )
        Rewriter.replace_op(function, sequence)


class TransformCodeGenerationVisitor(CodeGenerationVisitor):
    """Recognize the prototype's size tuple and MatmulOp class argument."""

    def visit_Tuple(self, node: ast.Tuple) -> None:
        if len(node.elts) != 3:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "expected a tuple of three tile sizes",
            )
        i, j, k = (self.visit_single_value(element) for element in node.elts)
        self.inserter.insert_op(SizesOp(i, j, k))

    def visit_Call(self, node: ast.Call) -> None:
        # The registered MatchOp constructor has a prototype-specific Python
        # signature: (root, MatmulOp). All other calls use normal PyAST lowering.
        source = (
            self.type_converter.globals.get(node.func.id)
            if isinstance(node.func, ast.Name)
            else None
        )
        if not callable(source) or (
            self.type_converter.function_registry.get_operation_constructor(source)
            is not transform.MatchOp
        ):
            return super().visit_Call(node)
        if len(node.args) != 2 or node.keywords:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "expected match(root, MatmulOp)",
            )

        class_expr = node.args[1]
        attributes: list[str] = []
        while isinstance(class_expr, ast.Attribute):
            attributes.append(class_expr.attr)
            class_expr = class_expr.value
        value: Any = None
        if isinstance(class_expr, ast.Name) and (
            self.symbol_table is None or class_expr.id not in self.symbol_table
        ):
            value = self.type_converter.globals.get(class_expr.id)
            for attribute in reversed(attributes):
                value = getattr(value, attribute, None)
        if value is not linalg.ops.MatmulOp:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "the Python transform prototype only supports matching MatmulOp",
            )
        root = self.visit_single_value(node.args[0])
        self.inserter.insert_op(transform.MatchOp(root, ops=[linalg.ops.MatmulOp.name]))


class PyTransformContext(PyASTContext):
    """
    Keep Python execution and lower the same function to a named sequence.

    Register matching helpers with transform.MatchOp and tiling helpers with
    TileOp. The generated module only contains the sequence;
    callers construct the outer transform.with_named_sequence module themselves.
    """

    def __init__(self):
        super().__init__(
            post_transforms=[FrontendDesymrefyPass(), LowerPyTransformPass()],
            code_generation_visitor=TransformCodeGenerationVisitor,
        )
        self.register_type(OperationHandle[Operation], transform.AnyOpType())
        self.register_literal(
            int, lambda value: arith.ConstantOp.from_int_and_width(value, 32)
        )
        self.register_dialect(PyTransform)


def apply_func(op: func.FuncOp, func: Callable[[OperationHandle], None]) -> func.FuncOp:
    clone = op.clone()
    func(OperationHandle(clone))
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

    TransformInterpreterPass().apply(ctx, schedule_module)
    schedule_module.verify()

    return transform_result.clone()


def _match(
    roots: OperationHandle[Operation], cls: type[OperationInvT]
) -> OperationHandle[OperationInvT]:
    matches = OperationHandle(
        *(
            candidate
            for candidate in roots.ops[0].walk(region_first=True)
            if isinstance(candidate, cls)
        )
    )
    return matches


def _tile(
    targets: OperationHandle[linalg.ops.MatmulOp], sizes: tuple[int, int, int]
) -> tuple[
    OperationHandle[linalg.ops.MatmulOp],
    OperationHandle[scf.ForOp],
    OperationHandle[scf.ForOp],
    OperationHandle[scf.ForOp],
]:
    for target in targets.ops:
        num_loops = target.get_num_loops()
        assert len(sizes) <= num_loops

    tiled_ops: list[linalg.ops.MatmulOp] = []
    loops: list[list[scf.ForOp]] = [[] for _ in sizes]
    for target in targets.ops:
        rewriter = PatternRewriter(target)
        result = tile_structured_op(rewriter, target, sizes)
        assert isinstance(result.tiled_op, linalg.ops.MatmulOp)
        tiled_ops.append(result.tiled_op)
        # Each transform result groups the corresponding loop across targets.
        for loop_handle, loop in zip(loops, result.loops, strict=True):
            loop_handle.append(loop)

    l0, l1, l2 = loops

    return (
        OperationHandle(*tiled_ops),
        OperationHandle(*l0),
        OperationHandle(*l1),
        OperationHandle(*l2),
    )


ctx = PyTransformContext()
ctx.register_function(_match, transform.MatchOp)
ctx.register_function(_tile, TileOp)


def print_intermediate(
    previous: ModulePass | None, module: builtin.ModuleOp, next_pass: ModulePass | None
) -> None:
    module.verify()
    if isinstance(previous, FrontendDesymrefyPass):
        print(module)


ctx.post_callback = print_intermediate


@ctx.parse_program
def tile(op: OperationHandle[Operation]) -> None:
    """Tile a structured linalg operation by 32 in each of its three dimensions."""
    targets = _match(op, linalg.ops.MatmulOp)
    _tile(targets, (32, 32, 32))


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
# Compilation leaves the sequence in a plain frontend module. The test builds
# the separate outer transform module in apply_transform.
sequence = tile.module.body.block.first_op
assert isinstance(sequence, transform.NamedSequenceOp)
print(tile.module)
transform_result = apply_transform(build_input(), sequence.clone())

assert python_result.is_structurally_equivalent(transform_result), (
    "Python execution and the compiled named sequence produced different IR"
)

print(python_result)


# Different constants must be read from the helper IR, not hardcoded to 32.
@ctx.parse_program
def different_sizes(op: OperationHandle[Operation]) -> None:
    targets = _match(op, linalg.ops.MatmulOp)
    _tile(targets, (16, 8, 4))


sequence = different_sizes.module.body.block.first_op
assert isinstance(sequence, transform.NamedSequenceOp)
print(different_sizes.module)
assert apply_func(build_input(), different_sizes).is_structurally_equivalent(
    apply_transform(build_input(), sequence.clone())
)


def print_compile_error(program: PyASTProgram[..., Any]) -> None:
    try:
        program.module
    except (CodeGenerationException, DiagnosticException) as error:
        print(error.msg if isinstance(error, CodeGenerationException) else str(error))
    else:
        raise AssertionError("expected a prototype diagnostic")


# Report unsupported forms instead of silently changing the Python schedule.
ctx.post_callback = None


@ctx.parse_program
def wrong_class(op: OperationHandle[Operation]) -> None:
    _match(op, scf.ForOp)


print_compile_error(wrong_class)


@ctx.parse_program
def wrong_size_count(op: OperationHandle[Operation]) -> None:
    targets = _match(op, linalg.ops.MatmulOp)
    _tile(targets, (32, 32))  # pyright: ignore[reportArgumentType]


print_compile_error(wrong_size_count)


@ctx.parse_program
def zero_size(op: OperationHandle[Operation]) -> None:
    targets = _match(op, linalg.ops.MatmulOp)
    _tile(targets, (32, 0, 32))


print_compile_error(zero_size)


def add_sizes(lhs: int, rhs: int) -> int:
    return lhs + rhs


ctx.register_function(add_sizes, arith.AddiOp)


@ctx.parse_program
def computed_size(op: OperationHandle[Operation]) -> None:
    targets = _match(op, linalg.ops.MatmulOp)
    _tile(targets, (add_sizes(16, 16), 32, 32))


print_compile_error(computed_size)

# CHECK:       builtin.module {
# CHECK-NEXT:    func.func @tile(%op: !transform.any_op) {
# CHECK-NEXT:      %0 = transform.structured.match ops{["linalg.matmul"]} in %op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = arith.constant 32 : i32
# CHECK-NEXT:      %2 = arith.constant 32 : i32
# CHECK-NEXT:      %3 = arith.constant 32 : i32
# CHECK-NEXT:      %4 = "pytransform.sizes"(%1, %2, %3) : (i32, i32, i32) -> tuple<i32, i32, i32>
# CHECK-NEXT:      %5, %6, %7, %8 = "pytransform.tile"(%0, %4) : (!transform.any_op, tuple<i32, i32, i32>) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      func.return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    transform.named_sequence @__transform_main(%op: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match ops{["linalg.matmul"]} in %op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1, %2, %3, %4 = transform.structured.tile_using_for %0 tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.yield
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
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
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    func.func @different_sizes(%op: !transform.any_op) {
# CHECK-NEXT:      %0 = transform.structured.match ops{["linalg.matmul"]} in %op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1 = arith.constant 16 : i32
# CHECK-NEXT:      %2 = arith.constant 8 : i32
# CHECK-NEXT:      %3 = arith.constant 4 : i32
# CHECK-NEXT:      %4 = "pytransform.sizes"(%1, %2, %3) : (i32, i32, i32) -> tuple<i32, i32, i32>
# CHECK-NEXT:      %5, %6, %7, %8 = "pytransform.tile"(%0, %4) : (!transform.any_op, tuple<i32, i32, i32>) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      func.return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    transform.named_sequence @__transform_main(%op: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match ops{["linalg.matmul"]} in %op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1, %2, %3, %4 = transform.structured.tile_using_for %0 tile_sizes [16, 8, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.yield
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  the Python transform prototype only supports matching MatmulOp
# CHECK-NEXT:  expected a tuple of three tile sizes
# CHECK-NEXT:  the Python transform prototype requires positive tile sizes
# CHECK-NEXT:  expected tile sizes from arith.constant i32 operations
