from pathlib import Path

from xdsl.ir import OpResult, BlockArgument, SSAValue
from xdsl.dialects.builtin import FunctionType, f64, ModuleOp
from xdsl.builder import Builder

from ..parser import Parser
from ..ir_gen import IRGen

from toy import dialect as toy


def test_convert_ast():
    ast_toy = Path('docs/Toy/examples/ast.toy')

    with open(ast_toy, 'r') as f:
        parser = Parser(ast_toy, f.read())

    module_ast = parser.parseModule()

    ir_gen = IRGen()

    generated_module_op = ir_gen.ir_gen_module(module_ast)

    @ModuleOp.from_region_or_ops
    @Builder.region
    def module_op(builder: Builder):
        unrankedf64TensorType = toy.UnrankedTensorType.from_type(f64)

        multiply_transpose_type = FunctionType.from_lists(
            [unrankedf64TensorType, unrankedf64TensorType],
            [unrankedf64TensorType])

        @Builder.callable_region(multiply_transpose_type.inputs)
        def multiply_transpose(builder: Builder, args: tuple[BlockArgument,
                                                             ...]) -> None:
            a, b = args
            a_t = builder.insert(toy.TransposeOp.from_input(a)).res
            b_t = builder.insert(toy.TransposeOp.from_input(b)).res
            prod = builder.insert(toy.MulOp.from_summands(a_t, b_t)).res
            builder.insert(toy.ReturnOp.from_input(prod))

        def call_multiply_transpose(builder: Builder, a: SSAValue,
                                    b: SSAValue) -> OpResult:
            return builder.insert(
                toy.GenericCallOp.get("multiply_transpose", [a, b],
                                      [unrankedf64TensorType])).res[0]

        main_type = FunctionType.from_lists([], [])

        @Builder.callable_region
        def main(builder: Builder) -> None:
            a = builder.insert(
                toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3])).res
            b_0 = builder.insert(
                toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6])).res
            b = builder.insert(toy.ReshapeOp.from_input(b_0, [2, 3])).res
            c = call_multiply_transpose(builder, a, b)
            call_multiply_transpose(builder, b, a)
            call_multiply_transpose(builder, b, c)
            a_t = builder.insert(toy.TransposeOp.from_input(a)).res
            call_multiply_transpose(builder, a_t, c)
            builder.insert(toy.ReturnOp.from_input())

        builder.insert(
            toy.FuncOp.from_region("multiply_transpose",
                                   multiply_transpose_type,
                                   multiply_transpose,
                                   private=True))
        builder.insert(toy.FuncOp.from_region("main", main_type, main))

    assert module_op.is_structurally_equivalent(generated_module_op)
