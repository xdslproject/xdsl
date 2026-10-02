# RUN: python %s | filecheck %s

from collections.abc import Callable
from typing import Any

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import builtin, func, get_all_dialects, linalg, py, scf, transform
from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
from xdsl.dialects.linalg.transforms.tiling import tile_structured_op
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.program import PyASTProgram
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.interpreters.transform import OperationHandle
from xdsl.ir import Dialect, Operation, OpResult, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import Rewriter
from xdsl.transforms.dead_code_elimination import region_dce
from xdsl.transforms.desymref import FrontendDesymrefyPass
from xdsl.transforms.linalg_generalize_named_ops import LinalgGeneralizeNamedOpsPass
from xdsl.transforms.transform_interpreter import TransformInterpreterPass
from xdsl.utils.exceptions import DiagnosticException, InterpretationError
from xdsl.utils.hints import isa
from xdsl.utils.lexer import Location


@irdl_op_definition
class MatchOp(IRDLOperation):
    """Match payload operations by a name resolved to a property during lowering."""

    name = "pytransform.match"

    root = operand_def(transform.AnyOpType)
    operation_name = operand_def(py.PyObjectType)
    result = result_def(transform.AnyOpType)

    def __init__(self, roots: SSAValue, name: SSAValue):
        super().__init__(operands=[roots, name], result_types=[transform.AnyOpType()])


@irdl_op_definition
class TileOp(IRDLOperation):
    """
    Tile a payload using sizes that will be resolved to properties by a pass.

    The prototype supports three positive constant sizes and returns four handles.
    This operation represents a payload mutation, even when all results are unused.
    """

    name = "pytransform.tile"

    target = operand_def(transform.AnyOpType)
    sizes = operand_def(py.PyObjectType)
    tiled = result_def(transform.AnyOpType)
    i = result_def(transform.AnyOpType)
    j = result_def(transform.AnyOpType)
    k = result_def(transform.AnyOpType)

    def __init__(self, targets: SSAValue, sizes: SSAValue):
        super().__init__(
            operands=[targets, sizes], result_types=[transform.AnyOpType()] * 4
        )


PyTransform = Dialect("pytransform", [MatchOp, TileOp], [])


class LowerPyTransformPass(ModulePass):
    """Lower one straight-line schedule with three positive constant tile sizes."""

    name = "lower-pytransform"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        functions = tuple(op.ops)
        if len(functions) != 1 or not isinstance(
            function := functions[0], transform.NamedSequenceOp
        ):
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
        if not isinstance(terminator, transform.YieldOp) or terminator.operands:
            raise DiagnosticException("expected a schedule ending in a bare return")

        for helper in tuple(block.ops):
            if isinstance(helper, MatchOp):
                if (
                    not isinstance(helper.operation_name, OpResult)
                    or not isinstance(
                        constant := helper.operation_name.owner, py.PyConstOp
                    )
                    or not isinstance(constant.const, builtin.StringAttr)
                ):
                    raise DiagnosticException("expected a constant operation name")
                Rewriter.replace_op(
                    helper, transform.MatchOp(helper.root, ops=[constant.const.data])
                )
                continue
            if not isinstance(helper, TileOp):
                continue
            if not isinstance(helper.sizes, OpResult) or not isinstance(
                sizes_op := helper.sizes.owner, py.PyBuildTupleOp
            ):
                raise DiagnosticException("expected tile sizes from py.build_tuple")
            if len(sizes_op.elements) != 3:
                raise DiagnosticException("expected a tuple of three tile sizes")
            sizes: list[int] = []
            for operand in sizes_op.operands:
                if (
                    not isinstance(operand, OpResult)
                    or not isinstance(constant := operand.owner, py.PyConstOp)
                    or not isinstance(constant.const, builtin.IntAttr)
                ):
                    raise DiagnosticException("expected constant integer tile sizes")
                size = constant.const.data
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
                body_op, transform.MatchOp | transform.TileOp | transform.YieldOp
            )
            for body_op in block.ops
        ):
            raise DiagnosticException(
                "unsupported operation in Python transform schedule"
            )


def build_named_sequence(
    name: str, signature: builtin.FunctionType, body: Region, location: Location
) -> transform.NamedSequenceOp:
    return transform.NamedSequenceOp(
        "__transform_main",
        signature,
        body,
        arg_attrs=[builtin.DictionaryAttr({"transform.readonly": builtin.UnitAttr()})],
    )


class PyTransformContext(PyASTContext):
    """
    Keep Python execution and lower the same function to a named sequence.

    Register matching helpers with MatchOp and tiling helpers with
    TileOp. The generated module only contains the sequence;
    callers construct the outer transform.with_named_sequence module themselves.
    """

    def __init__(self):
        super().__init__(
            post_transforms=[FrontendDesymrefyPass(), LowerPyTransformPass()],
        )
        self.register_function_definition(build_named_sequence)
        self.register_return(lambda values, location: transform.YieldOp(*values))
        self.register_type(OperationHandle[Operation], transform.AnyOpType())
        self.register_literal(int, lambda value: py.PyConstOp(builtin.IntAttr(value)))
        self.register_literal(
            str, lambda value: py.PyConstOp(builtin.StringAttr(value))
        )
        self.register_tuple(py.PyBuildTupleOp)
        self.register_dialect(py.Py)
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


def _match(roots: OperationHandle[Operation], name: str) -> OperationHandle[Operation]:
    matches = OperationHandle(
        *(
            candidate
            for candidate in roots.ops[0].walk(region_first=True)
            if candidate.name == name
        )
    )
    return matches


def _tile(
    targets: OperationHandle[Operation], sizes: tuple[int, int, int]
) -> tuple[
    OperationHandle[Operation],
    OperationHandle[scf.ForOp],
    OperationHandle[scf.ForOp],
    OperationHandle[scf.ForOp],
]:
    if not isa(targets.ops, tuple[LinalgStructuredOperation, ...]):
        raise ValueError("tiling requires structured linalg operations")
    for target in targets.ops:
        num_loops = target.get_num_loops()
        assert len(sizes) <= num_loops

    tiled_ops: list[Operation] = []
    loops: list[list[scf.ForOp]] = [[] for _ in sizes]
    for target in targets.ops:
        rewriter = PatternRewriter(target)
        result = tile_structured_op(rewriter, target, sizes)
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
ctx.register_function(_match, MatchOp)
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
    targets = _match(op, "linalg.matmul")
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


# Different constants must be read from the helper IR, not hardcoded to 32.
@ctx.parse_program
def different_sizes(op: OperationHandle[Operation]) -> None:
    targets = _match(op, "linalg.matmul")
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


ctx.post_callback = None


# Other structured operations work, including locally bound names and sizes.
@ctx.parse_program
def tile_generic(op: OperationHandle[Operation]) -> None:
    name = "linalg.generic"
    sizes = (16, 8, 4)
    targets = _match(op, name=name)
    _tile(targets=targets, sizes=sizes)


generic_input = build_input()
LinalgGeneralizeNamedOpsPass().apply(Context(), builtin.ModuleOp([generic_input]))
sequence = tile_generic.module.body.block.first_op
assert isinstance(sequence, transform.NamedSequenceOp)
assert apply_func(generic_input, tile_generic).is_structurally_equivalent(
    apply_transform(generic_input, sequence.clone())
)


# Report unsupported forms instead of silently changing the Python schedule.
@ctx.parse_program
def wrong_name(op: OperationHandle[Operation]) -> None:
    _match(op, 1)  # pyright: ignore[reportArgumentType]


print_compile_error(wrong_name)


# Matching is name-based; tiling checks the payload type in both execution paths.
@ctx.parse_program
def non_structured(op: OperationHandle[Operation]) -> None:
    targets = _match(op, "func.func")
    _tile(targets, (32, 32, 32))


try:
    apply_func(build_input(), non_structured)
except ValueError as error:
    print(error)
else:
    raise AssertionError("expected a structured operation diagnostic")

sequence = non_structured.module.body.block.first_op
assert isinstance(sequence, transform.NamedSequenceOp)
with pytest.raises(
    InterpretationError, match="supports only structured linalg targets"
):
    apply_transform(build_input(), sequence.clone())


@ctx.parse_program
def wrong_size_count(op: OperationHandle[Operation]) -> None:
    targets = _match(op, "linalg.matmul")
    _tile(targets, (32, 32))  # pyright: ignore[reportArgumentType]


print_compile_error(wrong_size_count)


@ctx.parse_program
def zero_size(op: OperationHandle[Operation]) -> None:
    targets = _match(op, "linalg.matmul")
    _tile(targets, (32, 0, 32))


print_compile_error(zero_size)


def add_sizes(lhs: int, rhs: int) -> int:
    return lhs + rhs


def build_add_sizes(lhs: SSAValue, rhs: SSAValue) -> py.PyBinOp:
    return py.PyBinOp(builtin.StringAttr("add"), lhs, rhs)


ctx.register_function(add_sizes, build_add_sizes)


@ctx.parse_program
def computed_size(op: OperationHandle[Operation]) -> None:
    targets = _match(op, "linalg.matmul")
    _tile(targets, (add_sizes(16, 16), 32, 32))


print_compile_error(computed_size)

# CHECK:       builtin.module {
# CHECK-NEXT:    transform.named_sequence @__transform_main(%op: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = py.const "linalg.matmul"
# CHECK-NEXT:      %1 = "pytransform.match"(%op, %0) : (!transform.any_op, !py.object) -> !transform.any_op
# CHECK-NEXT:      %2 = py.const 32
# CHECK-NEXT:      %3 = py.const 32
# CHECK-NEXT:      %4 = py.const 32
# CHECK-NEXT:      %5 = py.build_tuple(%2, %3, %4)
# CHECK-NEXT:      %6, %7, %8, %9 = "pytransform.tile"(%1, %5) : (!transform.any_op, !py.object) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.yield
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    transform.named_sequence @__transform_main(%op: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match ops{["linalg.matmul"]} in %op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1, %2, %3, %4 = transform.structured.tile_using_for %0 tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.yield
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    transform.named_sequence @__transform_main(%op: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = py.const "linalg.matmul"
# CHECK-NEXT:      %1 = "pytransform.match"(%op, %0) : (!transform.any_op, !py.object) -> !transform.any_op
# CHECK-NEXT:      %2 = py.const 16
# CHECK-NEXT:      %3 = py.const 8
# CHECK-NEXT:      %4 = py.const 4
# CHECK-NEXT:      %5 = py.build_tuple(%2, %3, %4)
# CHECK-NEXT:      %6, %7, %8, %9 = "pytransform.tile"(%1, %5) : (!transform.any_op, !py.object) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.yield
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    transform.named_sequence @__transform_main(%op: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match ops{["linalg.matmul"]} in %op : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %1, %2, %3, %4 = transform.structured.tile_using_for %0 tile_sizes [16, 8, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.yield
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  expected a constant operation name
# CHECK-NEXT:  tiling requires structured linalg operations
# CHECK-NEXT:  expected a tuple of three tile sizes
# CHECK-NEXT:  the Python transform prototype requires positive tile sizes
# CHECK-NEXT:  expected constant integer tile sizes
