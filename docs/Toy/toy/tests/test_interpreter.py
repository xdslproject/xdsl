from io import StringIO

from xdsl.ir import BlockArgument, Operation
from xdsl.dialects.builtin import f64, ModuleOp
from xdsl.interpreter import Interpreter

from ..dialects import toy as td
from ..interpreter import Tensor, ToyFunctions


def test_tensor_printing():
    assert f"{Tensor([], [0])}" == "[]"
    assert f"{Tensor([1], [1])}" == "[1]"
    assert f"{Tensor([1], [1, 1])}" == "[[1]]"
    assert f"{Tensor([1], [1, 1, 1])}" == "[[[1]]]"
    assert f"{Tensor([1, 2, 3, 4, 5, 6], [2, 3])}" == "[[1, 2, 3], [4, 5, 6]]"
    assert f"{Tensor([1, 2, 3, 4, 5, 6], [3, 2])}" == "[[1, 2], [3, 4], [5, 6]]"
