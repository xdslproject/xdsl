
from dataclasses import dataclass
from xdsl.builder import Builder
from xdsl.dialects.builtin import Float32Type, FunctionType, ModuleOp, TensorType
from jax._src.core import ClosedJaxpr

from xdsl.dialects.func import FuncOp, Return
from xdsl.ir.core import Block, Region


class IRGenError(Exception):
    pass


@dataclass(init=False)
class IRGen:
    """
    Implementation of a simple MLIR emission from the jaxpr.

    This will emit operations that are specific to the Jax language, preserving
    the semantics of the language and (hopefully) allow to perform accurate
    analysis and transformation based on these high level semantics.
    """

    module: ModuleOp
    """A "module" matches a jaxpr source file: containing a list of functions."""

    builder: Builder

    """
    The builder is a helper class to create IR inside a function. The builder
    is stateful, in particular it keeps an "insertion point": this is where
    the next operations will be introduced."""

    def __init__(self):
        # We create an empty MLIR module and codegen functions one at a time and
        # add them to the module.
        self.module = ModuleOp([])
        self.builder = Builder.at_end(self.module.body.blocks[0])



    def ir_gen_module(self, module_ast: ClosedJaxpr) -> ModuleOp:
        """
        Public API: convert the jaxpr (source file) to an MLIR
        Module operation."""
        parent_builder = self.builder

        assert len(module_ast.jaxpr.invars) == 1
        assert len(module_ast.jaxpr.outvars) == 1

        inputVars = module_ast.jaxpr.invars[0]
        outputVars = module_ast.jaxpr.outvars[0]

        input_types = [TensorType(Float32Type(), [inputVars.aval.size]) for _ in range(len(module_ast.jaxpr.invars))]
        output_types = [TensorType(Float32Type(), [outputVars.aval.size]) for _ in range(len(module_ast.jaxpr.outvars))]

        block = Block(arg_types=[TensorType(Float32Type(), [inputVars.aval.size]) for _ in range(len(module_ast.jaxpr.invars))])
        self.builder = Builder.at_end(block)

        func_type = FunctionType.from_lists(input_types, output_types)
        # print(func_type)

        for eqn in module_ast.jaxpr.eqns:
            raise NotImplementedError(
                f"jax equation not implemented"
            )

        if (inputVars == outputVars):
            return_ssa = block.args[0]
            self.builder.insert(Return(return_ssa))
        else: 
            self.builder.insert(Return())
        self.builder = parent_builder

        func = self.builder.insert(
            FuncOp("main", func_type, Region(block), visibility='public'))
        print(func.args)
        print(func)

        return self.module
    
