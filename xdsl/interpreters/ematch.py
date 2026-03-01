from dataclasses import dataclass
from typing import Any

from xdsl.dialects import ematch, equivalence
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.ir import OpResult, SSAValue


@register_impls
@dataclass
class EmatchFunctions(InterpreterFunctions):
    """Interpreter functions for PDL patterns operating on e-graphs."""

    @impl(ematch.GetClassValsOp)
    def run_get_class_vals(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassValsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Take a value and return all values in its equivalence class.

        If the value is an equivalence.class result, return the operands of the class,
        otherwise return a tuple containing just the value itself.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return ((val,),)

        assert isinstance(val, SSAValue)

        if isinstance(val, OpResult):
            defining_op = val.owner
            if isinstance(defining_op, equivalence.AnyClassOp):
                # Find the leader to get the canonical set of operands
                leader = self.eclass_union_find.find(defining_op)
                return (tuple(leader.operands),)

        # Value is not an eclass result, return it as a single-element tuple
        return ((val,),)
