from io import StringIO
from xdsl.builder import Builder

from xdsl.dialects.builtin import (
    FunctionType,
    ModuleOp,
)

from xdsl.printer import Printer


from ..dialects import toy

# ctx = context()

toy_program = """
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<3, 2> = [1, 2, 3, 4, 5, 6];

  # There is a built-in print instruction to display the contents of the tensor
  print(b);

  # Reshapes are implicit on assignment
  var c<2, 3> = b;

  # There are + and * operators for pointwise addition and multiplication
  var d = a + c;

  print(d);
}
"""


def desc(op: ModuleOp) -> str:
    stream = StringIO()
    Printer(stream=stream).print(op)
    return stream.getvalue()


@ModuleOp
@Builder.implicit_region
def toy_0():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        b_0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6]).res
        b = toy.ReshapeOp(b_0, [3, 2]).res
        toy.PrintOp(b)
        c = toy.ReshapeOp(b, [2, 3]).res
        d = toy.AddOp(a, c).res
        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def toy_1():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        b = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [3, 2]).res
        toy.PrintOp(b)
        c = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        d = toy.AddOp(a, c).res
        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)
