from pathlib import Path

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects.builtin import FunctionType, ModuleOp, f64
from xdsl.ir import OpResult, SSAValue

from ..dialects import toy
from ..frontend.ir_gen import IRGen
from ..frontend.parser import ToyParser


def test_convert_ast():
    ast_toy = Path("docs/Toy/examples/ast.toy")

    with open(ast_toy) as f:
        parser = ToyParser(ast_toy, f.read())

    module_ast = parser.parseModule()

    ir_gen = IRGen()

    generated_module_op = ir_gen.ir_gen_module(module_ast)

    @ModuleOp
    @Builder.implicit_region
    def module_op():
        unrankedf64TensorType = toy.UnrankedTensorType(f64)

        multiply_transpose_type = FunctionType.from_lists(
            [unrankedf64TensorType, unrankedf64TensorType], [unrankedf64TensorType]
        )

        with ImplicitBuilder(
            toy.FuncOp("multiply_transpose", multiply_transpose_type, private=True).body
        ) as (a, b):
            a_t = toy.TransposeOp(a).res
            b_t = toy.TransposeOp(b).res
            prod = toy.MulOp(a_t, b_t).res
            toy.ReturnOp(prod)

        def call_multiply_transpose(a: SSAValue, b: SSAValue) -> OpResult:
            return toy.GenericCallOp(
                "multiply_transpose", [a, b], [unrankedf64TensorType]
            ).res[0]

        main_type = FunctionType.from_lists([], [])

        with ImplicitBuilder(toy.FuncOp("main", main_type).body):
            a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
            b_0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6]).res
            b = toy.ReshapeOp(b_0, [2, 3]).res
            c = call_multiply_transpose(a, b)
            call_multiply_transpose(b, a)
            call_multiply_transpose(b, c)
            a_t = toy.TransposeOp(a).res
            call_multiply_transpose(a_t, c)
            toy.ReturnOp()

    assert module_op.is_structurally_equivalent(generated_module_op)


def test_convert_scalar():
    scalar_toy = Path("docs/Toy/examples/scalar.toy")

    with open(scalar_toy) as f:
        parser = ToyParser(scalar_toy, f.read())

    module_ast = parser.parseModule()

    ir_gen = IRGen()

    generated_module_op = ir_gen.ir_gen_module(module_ast)

    @ModuleOp
    @Builder.implicit_region
    def module_op():
        with ImplicitBuilder(toy.FuncOp("main", FunctionType.from_lists([], [])).body):
            a_0 = toy.ConstantOp.from_value(5.5).res
            a = toy.ReshapeOp(a_0, [2, 2]).res
            toy.PrintOp(a)
            toy.ReturnOp()

    assert module_op.is_structurally_equivalent(generated_module_op)
