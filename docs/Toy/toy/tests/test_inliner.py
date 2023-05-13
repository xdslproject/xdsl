from xdsl.builder import Builder
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.ir import BlockArgument, OpResult, SSAValue


from ..compiler import context
from ..dialects import toy
from ..rewrites.inline_toy import InlineToyPass
from ..rewrites.optimise_toy import OptimiseToy


@ModuleOp
@Builder.implicit_region
def toy_0():
    unrankedf64TensorType = toy.UnrankedTensorType.from_type(toy.f64)

    multiply_transpose_type = FunctionType.from_lists(
        [unrankedf64TensorType, unrankedf64TensorType], [unrankedf64TensorType]
    )

    @Builder.implicit_region(multiply_transpose_type.inputs)
    def multiply_transpose(args: tuple[BlockArgument, ...]) -> None:
        a, b = args
        a_t = toy.TransposeOp(a).res
        b_t = toy.TransposeOp(b).res
        prod = toy.MulOp(a_t, b_t).res
        toy.ReturnOp(prod)

    def call_multiply_transpose(a: SSAValue, b: SSAValue) -> OpResult:
        return toy.GenericCallOp(
            "multiply_transpose", [a, b], [unrankedf64TensorType]
        ).res[0]

    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        x0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        x1 = toy.ReshapeOp(x0, [2, 3]).res
        x2 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6]).res
        x3 = toy.ReshapeOp(x2, [2, 3]).res
        _x4 = call_multiply_transpose(x1, x3)
        x5 = call_multiply_transpose(x3, x1)
        toy.PrintOp(x5)
        toy.ReturnOp()

    toy.FuncOp(
        "multiply_transpose",
        multiply_transpose_type,
        multiply_transpose,
        private=True,
    )
    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def toy_1():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        x0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        x1 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        x2 = toy.CastOp(x1).res
        x3 = toy.CastOp(x0).res
        x4 = toy.TransposeOp(x2).res
        x5 = toy.TransposeOp(x3).res
        x6 = toy.MulOp(x4, x5).res
        toy.PrintOp(x6)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


ctx = context()


def test_inline():
    copy = toy_0.clone()
    OptimiseToy().apply(ctx, copy)
    InlineToyPass().apply(ctx, copy)
    assert str(copy) == str(toy_1)
