from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects import ematch, equivalence
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, OpResult, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.utils.disjoint_set import DisjointSet
from xdsl.utils.hints import isa


@register_impls
@dataclass
class EmatchFunctions(InterpreterFunctions):
    """Interpreter functions for PDL patterns operating on e-graphs."""

    eclass_union_find: DisjointSet[equivalence.AnyClassOp] = field(
        default_factory=lambda: DisjointSet[equivalence.AnyClassOp]()
    )
    """Union-find structure tracking which e-classes are equivalent and should be merged."""

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
                return (tuple(defining_op.operands),)

        # Value is not an eclass result, return it as a single-element tuple
        return ((val,),)

    @impl(ematch.GetClassRepresentativeOp)
    def run_get_class_representative(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassRepresentativeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get one of the values in the equivalence class of v.
        Returns the first operand of the equivalence class.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return (val,)

        assert isa(val, SSAValue)

        if isinstance(val, OpResult):
            defining_op = val.owner
            if isinstance(defining_op, equivalence.AnyClassOp):
                return (defining_op.operands[0],)

        # Value is not an eclass result, return it as-is
        return (val,)

    @impl(ematch.GetClassResultOp)
    def run_get_class_result(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassResultOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get the equivalence.class result corresponding to the equivalence class of v.

        If v has exactly one use and that use is a ClassOp, return the ClassOp's result.
        Otherwise return v unchanged.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return (val,)

        assert isa(val, SSAValue)

        if val.has_one_use():
            user = val.get_user_of_unique_use()
            if isinstance(user, equivalence.AnyClassOp):
                return (user.result,)

        return (val,)

    @impl(ematch.GetClassResultsOp)
    def run_get_class_results(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassResultsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get the equivalence.class results corresponding to the equivalence classes
        of a range of values.
        """
        assert len(args) == 1
        vals = args[0]

        if vals is None:
            return ((),)

        results: list[SSAValue] = []
        for val in vals:
            if val is None:
                results.append(val)
            elif val.has_one_use():
                user = val.get_user_of_unique_use()
                if isinstance(user, equivalence.AnyClassOp):
                    results.append(user.result)
                else:
                    results.append(val)
            else:
                results.append(val)

        return (tuple(results),)

    def get_or_create_class(
        self, interpreter: Interpreter, val: SSAValue
    ) -> equivalence.AnyClassOp:
        """
        Get the equivalence class for a value, creating one if it doesn't exist.
        """
        if isinstance(val, OpResult):
            # If val is defined by a ClassOp, return it
            if isinstance(val.owner, equivalence.AnyClassOp):
                return self.eclass_union_find.find(val.owner)
            insertpoint = InsertPoint.before(val.owner)
        else:
            assert isinstance(val.owner, Block)
            insertpoint = InsertPoint.at_start(val.owner)

        # If val has one use and it's a ClassOp, return it
        if (user := val.get_user_of_unique_use()) is not None:
            if isinstance(user, equivalence.AnyClassOp):
                return user

        # If the value is not part of an eclass yet, create one
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)

        eclass_op = equivalence.ClassOp(val)
        rewriter.insert_op(eclass_op, insertpoint)
        self.eclass_union_find.add(eclass_op)

        # Replace uses of val with the eclass result (except in the eclass itself)
        rewriter.replace_uses_with_if(
            val, eclass_op.result, lambda use: use.operation is not eclass_op
        )

        return eclass_op
