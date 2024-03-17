from typing import Any

from xdsl.dialects import ml_program
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)


@register_impls
class MLProgramFunctions(InterpreterFunctions):
    @impl(ml_program.Global)
    def run_global(
        self, interpreter: Interpreter, op: ml_program.Global, args: tuple[Any, ...]
    ):
        pass

    @impl(ml_program.GlobalLoadConstant)
    def run_global(
        self,
        interpreter: Interpreter,
        op: ml_program.GlobalLoadConstant,
        args: tuple[Any, ...],
    ):
        pass
