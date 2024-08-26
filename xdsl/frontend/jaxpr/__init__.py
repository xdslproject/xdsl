from dataclasses import dataclass, field
from typing import NoReturn

from jax._src.core import ClosedJaxpr

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, f32
from xdsl.dialects.func import FuncOp, Return
from xdsl.ir import Block, Region, SSAValue

from .primitive_builder import PrimitiveBuilder


class IRGenError(Exception):
    pass


@dataclass
class ScopedSymbolTable:
    "A mapping from variable names to SSAValues, append-only"

    table: dict[str, SSAValue] = field(default_factory=dict)

    def __contains__(self, __o: object) -> bool:
        return __o in self.table

    def __getitem__(self, __key: str) -> SSAValue:
        return self.table[__key]

    def __setitem__(self, __key: str, __value: SSAValue) -> None:
        if __key in self:
            raise AssertionError(f"Cannot add value for key {__key} in scope {self}")
        self.table[__key] = __value


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
    primitive_builder_jax: PrimitiveBuilder

    """
    The builder is a helper class to create IR inside a function. The builder
    is stateful, in particular it keeps an "insertion point": this is where
    the next operations will be introduced."""

    symbol_table: ScopedSymbolTable | None = None
    """
    The symbol table maps a variable name to a value in the current scope.
    Entering a function creates a new scope, and the function arguments are
    added to the mapping. When the processing of a function is terminated, the
    scope is destroyed and the mappings created in this scope are dropped."""

    def __init__(self):
        # We create an empty MLIR module and codegen functions one at a time and
        # add them to the module.
        self.module = ModuleOp([])
        self.primitive_builder_jax = PrimitiveBuilder()
        self.builder = Builder.at_end(self.module.body.blocks[0])

    def declare(self, var: str, value: SSAValue) -> bool:
        """
        Declare a variable in the current scope, return success if the variable
        wasn't declared yet."""
        assert self.symbol_table is not None
        if var in self.symbol_table:
            return False
        self.symbol_table[var] = value
        return True

    def ir_gen_variable_expr(self, name: str) -> SSAValue:  # pyright: ignore[reportReturnType]
        """
        This is a reference to a variable in an expression. The variable is
        expected to have been declared and so should have a value in the symbol
        table, otherwise emit an error and return nullptr."""
        assert self.symbol_table is not None
        try:
            variable = self.symbol_table[name]
            return variable
        except Exception as e:
            self.error(f"error: unknown variable `{name}`", e)

    def insert_return_op(self, return_vars: list[str]) -> None:
        assert len(return_vars) == 1
        with ImplicitBuilder(self.builder):
            return_ssa = self.ir_gen_variable_expr(return_vars[0])
            Return(return_ssa)

    def ir_gen_module(self, jaxpr: ClosedJaxpr) -> ModuleOp:
        """
        Public API: convert the jaxpr (source file) to an MLIR
        Module operation."""
        parent_builder = self.builder
        self.symbol_table = ScopedSymbolTable()

        assert len(jaxpr.jaxpr.outvars) == 1

        for arg in jaxpr.jaxpr.invars:
            assert arg.aval.dtype == "float32"

        inputVars = jaxpr.jaxpr.invars
        outputVars = jaxpr.jaxpr.outvars

        for i in range(1, len(inputVars)):
            assert inputVars[i].aval.shape == inputVars[i - 1].aval.shape

        input_types = [
            TensorType(f32, inputVars[0].aval.shape)
            for _ in range(len(jaxpr.jaxpr.invars))
        ]

        output_types = [
            TensorType(f32, outputVars[0].aval.shape)
            for _ in range(len(jaxpr.jaxpr.outvars))
        ]

        block = Block(arg_types=input_types)
        self.builder = Builder.at_end(block)

        for name, value in zip(inputVars, block.args):
            self.declare(name, value)

        func_type = FunctionType.from_lists(input_types, output_types)

        for eq in jaxpr.jaxpr.eqns:
            with ImplicitBuilder(self.builder):
                args = tuple(self.ir_gen_variable_expr(invar) for invar in eq.invars)
                res = self.primitive_builder_jax.build(eq, args)
                for outvar, result in zip(eq.outvars, res.results, strict=True):
                    self.declare(outvar, result)

        self.insert_return_op(outputVars)  # pyright: ignore[reportUnknownMemberType]

        self.builder = parent_builder
        with ImplicitBuilder(self.builder):
            FuncOp("main", func_type, Region(block), visibility="public")

        return self.module

    def error(self, message: str, cause: Exception | None = None) -> NoReturn:
        raise IRGenError(message) from cause
