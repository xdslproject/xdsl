from xdsl.builder import Builder
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.ir import BlockArgument, OpResult, SSAValue


from ..compiler import context
from ..dialects import toy
from ..rewrites.inline_toy import InlineToyPass
from ..rewrites.optimise_toy import OptimiseToy

"""
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %5 : tensor<*xf64>
  toy.return
}
"""


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


"""
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
  %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
  %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
  %6 = toy.mul %4, %5 : tensor<*xf64>
  toy.print %6 : tensor<*xf64>
  toy.return
}
"""


@ModuleOp
@Builder.implicit_region
def toy_1():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        x0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        x1 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [3, 2]).res
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
