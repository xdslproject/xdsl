from pathlib import Path

from xdsl.ir import OpResult, BlockArgument, SSAValue
from xdsl.dialects.builtin import f64, ModuleOp
from xdsl.builder import OpBuilder

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
    @OpBuilder.region
    def module_op(builder: OpBuilder):
        unrankedf64TensorType = toy.UnrankedTensorType.from_type(f64)

        @OpBuilder.callable_region(
            ([unrankedf64TensorType,
              unrankedf64TensorType], [unrankedf64TensorType]))
        def multiply_transpose(builder: OpBuilder, args: tuple[BlockArgument,
                                                               ...]) -> None:
            a, b = args
            a_t = builder.create(toy.TransposeOp.from_input, a).res
            b_t = builder.create(toy.TransposeOp.from_input, b).res
            prod = builder.create(toy.MulOp.from_summands, a_t, b_t).res
            builder.create(toy.ReturnOp.from_input, prod)

        def call_multiply_transpose(builder: OpBuilder, a: SSAValue,
                                    b: SSAValue) -> OpResult:
            return builder.create(toy.GenericCallOp.get, "multiply_transpose",
                                  [a, b], [unrankedf64TensorType]).res[0]

        @OpBuilder.callable_region
        def main(builder: OpBuilder) -> None:
            a = builder.create(toy.ConstantOp.from_list, [1, 2, 3, 4, 5, 6],
                               [2, 3]).res
            b_0 = builder.create(toy.ConstantOp.from_list, [1, 2, 3, 4, 5, 6],
                                 [6]).res
            b = builder.create(toy.ReshapeOp.from_input, b_0, [2, 3]).res
            c = call_multiply_transpose(builder, a, b)
            call_multiply_transpose(builder, b, a)
            call_multiply_transpose(builder, b, c)
            a_t = builder.create(toy.TransposeOp.from_input, a).res
            call_multiply_transpose(builder, a_t, c)
            builder.create(toy.ReturnOp.from_input)

        builder.create(toy.FuncOp.from_callable_region,
                       "multiply_transpose",
                       multiply_transpose,
                       private=True)
        builder.create(toy.FuncOp.from_callable_region, "main", main)

    assert module_op.is_structurally_equivalent(generated_module_op)
