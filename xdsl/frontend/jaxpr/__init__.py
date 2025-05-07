from dataclasses import dataclass

from jax._src.core import ClosedJaxpr

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, f32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region


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
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def ir_gen_module(self, jaxpr: ClosedJaxpr) -> ModuleOp:
        """
        Public API: convert the jaxpr (source file) to an MLIR
        Module operation."""
        parent_builder = self.builder

        assert len(jaxpr.jaxpr.invars) == 1
        assert len(jaxpr.jaxpr.outvars) == 1
        for arg in jaxpr.jaxpr.invars:
            assert arg.aval.dtype == "float32"

        inputVars = jaxpr.jaxpr.invars[0]
        outputVars = jaxpr.jaxpr.outvars[0]

        input_types = [
            TensorType(f32, [inputVars.aval.size])
            for _ in range(len(jaxpr.jaxpr.invars))
        ]
        output_types = [
            TensorType(f32, [outputVars.aval.size])
            for _ in range(len(jaxpr.jaxpr.outvars))
        ]

        block = Block(arg_types=input_types)
        self.builder = Builder(InsertPoint.at_end(block))

        func_type = FunctionType.from_lists(input_types, output_types)

        for _ in jaxpr.jaxpr.eqns:
            raise NotImplementedError("jax equation not implemented")

        assert inputVars == outputVars

        return_ssa = block.args[0]
        self.builder.insert(ReturnOp(return_ssa))

        self.builder = parent_builder

        self.builder.insert(
            FuncOp("main", func_type, Region(block), visibility="public")
        )

        return self.module
