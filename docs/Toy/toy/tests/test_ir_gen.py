from pathlib import Path

from xdsl.ir import OpResult, BlockArgument, SSAValue
from xdsl.dialects.builtin import FunctionType, f64, ModuleOp
from xdsl.builder import Builder

from ..parser import Parser
from ..ir_gen import IRGen

from toy import dialect as toy


def test_convert_ast():
    ast_toy = Path("docs/Toy/examples/ast.toy")

    with open(ast_toy, "r") as f:
        parser = Parser(ast_toy, f.read())

    module_ast = parser.parseModule()

    ir_gen = IRGen()

    generated_module_op = ir_gen.ir_gen_module(module_ast)

    @ModuleOp.from_region_or_ops
    @Builder.implicit_region
    def module_op():
        unrankedf64TensorType = toy.UnrankedTensorType.from_type(f64)

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
            a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
            b_0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6]).res
            b = toy.ReshapeOp(b_0, [2, 3]).res
            c = call_multiply_transpose(a, b)
            call_multiply_transpose(b, a)
            call_multiply_transpose(b, c)
            a_t = toy.TransposeOp(a).res
            call_multiply_transpose(a_t, c)
            toy.ReturnOp()

        toy.FuncOp(
            "multiply_transpose",
            multiply_transpose_type,
            multiply_transpose,
            private=True,
        )
        toy.FuncOp("main", main_type, main)

    assert module_op.is_structurally_equivalent(generated_module_op)
