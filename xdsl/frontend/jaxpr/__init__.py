
from dataclasses import dataclass
from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp
from jax._src.core import ClosedJaxpr


class IRGenError(Exception):
    pass


@dataclass(init=False)
class IRGen:
    """
    Implementation of a simple MLIR emission from the Toy AST.

    This will emit operations that are specific to the Toy language, preserving
    the semantics of the language and (hopefully) allow to perform accurate
    analysis and transformation based on these high level semantics.
    """

    module: ModuleOp
    """A "module" matches a Toy source file: containing a list of functions."""

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
        Public API: convert the AST for a Toy module (source file) to an MLIR
        Module operation."""

        assert False
