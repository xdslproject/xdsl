from io import StringIO
from xdsl.ir import BlockArgument, Operation
from xdsl.dialects.builtin import f64, ModuleOp

from .. import dialect as td
from ..interpreter import Tensor, execute_toy_module


def test_tensor_printing():
    assert f'{Tensor([], [0])}' == '[]'
    assert f'{Tensor([1], [1])}' == '[1]'
    assert f'{Tensor([1], [1, 1])}' == '[[1]]'
    assert f'{Tensor([1], [1, 1, 1])}' == '[[[1]]]'
    assert f'{Tensor([1, 2, 3, 4, 5, 6], [2, 3])}' == '[[1, 2, 3], [4, 5, 6]]'
    assert f'{Tensor([1, 2, 3, 4, 5, 6], [3, 2])}' == '[[1, 2], [3, 4], [5, 6]]'


def test_interpreter():
    module_op = build_module()
    stream = StringIO()
    execute_toy_module(module_op, file=stream)
    assert '[[1.0, 9.0], [25.0, 4.0], [16.0, 36.0]]\n' == stream.getvalue()


def build_module() -> ModuleOp:

    unrankedf64TensorType = td.UnrankedTensorType.from_type(f64)

    def func_body(*args: BlockArgument) -> list[Operation]:
        arg0, arg1 = args
        f0 = td.TransposeOp.from_input(arg0)
        f1 = td.TransposeOp.from_input(arg1)
        f2 = td.MulOp.from_summands(f0.results[0], f1.results[0])
        f3 = td.ReturnOp.from_input(f2.results[0])
        return [f0, f1, f2, f3]

    def main_body(*args: BlockArgument) -> list[Operation]:
        m0 = td.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3])
        [a] = m0.results
        m1 = td.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6])
        m2 = td.ReshapeOp.from_input(m1.results[0], [2, 3])
        [b] = m2.results
        m3 = td.GenericCallOp.get('multiply_transpose', [a, b],
                                  [unrankedf64TensorType])
        [c] = m3.results
        m4 = td.PrintOp.from_input(c)
        m5 = td.ReturnOp.from_input()
        return [m0, m1, m2, m3, m4, m5]

    multiply_transpose = td.FuncOp.from_callable(
        'multiply_transpose', [unrankedf64TensorType, unrankedf64TensorType],
        [unrankedf64TensorType],
        func_body,
        private=True)
    main = td.FuncOp.from_callable('main', [], [], main_body, private=False)

    return ModuleOp.from_region_or_ops([multiply_transpose, main])