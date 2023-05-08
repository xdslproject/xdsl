from pathlib import Path

from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp

from .frontend.ir_gen import IRGen
from .frontend.parser import Parser


def parse_toy(program: str, ctx: MLContext | None = None) -> ModuleOp:
    mlir_gen = IRGen()
    module_ast = Parser(Path("in_memory"), program).parseModule()
    module_op = mlir_gen.ir_gen_module(module_ast)
    return module_op
