from pathlib import Path

from xdsl.ir import Operation, BlockArgument
from xdsl.dialects.builtin import f64, ModuleOp

from ..parser import Parser
from ..mlir_gen import MLIRGen

from toy import dialect as td


def test_convert_ast():
    ast_toy = Path('docs/Toy/examples/ast.toy')

    with open(ast_toy, 'r') as f:
        parser = Parser(ast_toy, f.read())

    module_ast = parser.parseModule()

    mlir_gen = MLIRGen()

    generated_module_op = mlir_gen.mlir_gen_module(module_ast)

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
        m4 = td.GenericCallOp.get('multiply_transpose', [b, a],
                                  [unrankedf64TensorType])
        m5 = td.GenericCallOp.get('multiply_transpose', [b, c],
                                  [unrankedf64TensorType])
        m6 = td.TransposeOp.from_input(a)
        [a_transposed] = m6.results
        m7 = td.GenericCallOp.get('multiply_transpose', [a_transposed, c],
                                  [unrankedf64TensorType])
        m8 = td.ReturnOp.from_input()
        return [m0, m1, m2, m3, m4, m5, m6, m7, m8]

    multiply_transpose = td.FuncOp.from_callable(
        'multiply_transpose', [unrankedf64TensorType, unrankedf64TensorType],
        [unrankedf64TensorType],
        func_body,
        private=True)
    main = td.FuncOp.from_callable('main', [], [], main_body, private=False)

    module_op = ModuleOp.from_region_or_ops([multiply_transpose, main])

    assert module_op.is_structurally_equivalent(generated_module_op)
