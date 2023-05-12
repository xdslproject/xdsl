from io import StringIO

from xdsl.builder import Builder

from xdsl.dialects.builtin import (
    FunctionType,
    ModuleOp,
)

from xdsl.printer import Printer

from ..dialects import toy
from ..rewrites.shape_inference import ShapeInferencePass
from ..compiler import context

# ctx = context()


def desc(op: ModuleOp) -> str:
    stream = StringIO()
    Printer(stream=stream).print(op)
    return stream.getvalue()


"""
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
  %3 = toy.mul %2, %2 : tensor<*xf64>
  toy.print %3 : tensor<*xf64>
  toy.return
}
"""


@ModuleOp
@Builder.implicit_region
def toy_0():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        b = toy.CastOp(a).res
        c = toy.TransposeOp(b).res
        d = toy.MulOp(c, c).res
        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


"""
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
"""


@ModuleOp
@Builder.implicit_region
def toy_1():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        b = toy.TransposeOp(a).res
        c = toy.MulOp(b, b).res
        toy.PrintOp(c)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


ctx = context()


def test_optimise_toy():
    copy = toy_0.clone()
    ShapeInferencePass().apply(ctx, copy)
    assert desc(toy_1) == desc(copy)
    assert toy_1.is_structurally_equivalent(copy)
