from pathlib import Path

from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.ir import MLContext

from .dialects import toy
from .frontend.ir_gen import IRGen
from .frontend.parser import Parser
from .rewrites.inline_toy import InlineToyPass
from .rewrites.lower_toy_affine import LowerToAffinePass
from .rewrites.optimise_toy import OptimiseToy
from .rewrites.shape_inference import ShapeInferencePass


def context() -> MLContext:
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(toy.Toy)
    return ctx


def parse_toy(program: str, ctx: MLContext | None = None) -> ModuleOp:
    mlir_gen = IRGen()
    module_ast = Parser(Path("in_memory"), program).parseModule()
    module_op = mlir_gen.ir_gen_module(module_ast)
    return module_op


def transform(ctx: MLContext, module_op: ModuleOp, *, target: str = "toy-infer-shapes"):
    if target == "toy":
        return

    OptimiseToy().apply(ctx, module_op)

    if target == "toy-opt":
        return

    InlineToyPass().apply(ctx, module_op)

    if target == "toy-inline":
        return

    ShapeInferencePass().apply(ctx, module_op)

    if target == "toy-infer-shapes":
        return

    LowerToAffinePass().apply(ctx, module_op)
    module_op.verify()

    if target == "affine":
        return

    raise ValueError(f"Unknown target option {target}")
