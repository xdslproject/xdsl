from xdsl.ir import MLContext, SSAValue, Operation
from dialect import *
from xdsl.dialects.builtin import ModuleOp, UnrankedTensorType, f64
from typing import List

from xdsl.printer import Printer

# MLContext, containing information about the registered dialects
context = MLContext()

toy = Toy(context)


def func_body(arg0: SSAValue, arg1: SSAValue) -> List[Operation]:
    f0 = TransposeOp.from_input(arg0)
    f1 = TransposeOp.from_input(arg1)
    f2 = MulOp.from_summands(f0.results[0], f1.results[0])
    f3 = ReturnOp.from_input(f2.results[0])
    return [f0, f1, f2, f3]

unrankedTensorTypeF64 = UnrankedTensorType.from_type(f64)

def main_body() -> List[Operation]:
    m0 = ConstantOp.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    m1 = ReshapeOp.from_input(m0.results[0], [2, 3])
    m2 = ConstantOp.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6])
    m3 = ReshapeOp.from_input(m2.results[0], [2, 3])
    m4 = GenericCallOp.get('multiply_transpose',
                           [m1.results[0], m3.results[0]],
                           [unrankedTensorTypeF64])
    m5 = GenericCallOp.get('multiply_transpose',
                           [m3.results[0], m1.results[0]],
                           [unrankedTensorTypeF64])
    m6 = PrintOp.from_input(m5.results[0])
    m7 = ReturnOp.from_input()
    return [m0, m1, m2, m3, m4, m5, m6, m7]


multiply_transpose = FuncOp.from_callable(
    'multiply_transpose', [unrankedTensorTypeF64, unrankedTensorTypeF64],
    [unrankedTensorTypeF64],
    func_body,
    private=True)
main = FuncOp.from_callable('main', [], [], main_body)

module = ModuleOp.from_region_or_ops([multiply_transpose, main])

from pathlib import Path

print(Path().absolute())

with open('docs/toy/codegen-xdsl.mlir', 'w') as f:
    # Printer used to pretty-print MLIR data structures
    printer = Printer(target=Printer.Target.MLIR, stream=f)
    printer.print(module)
