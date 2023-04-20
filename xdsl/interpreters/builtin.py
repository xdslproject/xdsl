from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, cast

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.ir.core import Attribute
from xdsl.utils.exceptions import InterpretationError

_AttributeT0 = TypeVar("_AttributeT0")
_AttributeT1 = TypeVar("_AttributeT1")


@dataclass
@register_impls
class BuiltinFunctions(InterpreterFunctions):
    cast_impls: dict[
        tuple[type[Attribute], type[Attribute]],
        Callable[[Attribute, Attribute, Any], Any],
    ] = field(default_factory=lambda: {})

    def register_cast_impl(
        self,
        o_t: type[_AttributeT0],
        r_t: type[_AttributeT1],
        impl: Callable[[_AttributeT0, _AttributeT1, Any], Any],
    ) -> None:
        # Assume that the wrong Attribute type can never be passed
        impl_ = cast(Callable[[Attribute, Attribute, Any], Any], impl)
        self.cast_impls[(o_t, r_t)] = impl_

    def cast_value(self, o: Attribute, r: Attribute, value: Any) -> Any:
        """
        If the type of the operand and result are not the same, then look up the
        user-provided conversion function.
        """
        if o == r:
            return value

        fn = self.cast_impls.get((type(o), type(r)))
        if fn is None:
            raise InterpretationError(f"Cannot cast value {value} of type {o} to {r}")

        return fn(o, r, value)

    @impl(UnrealizedConversionCastOp)
    def run_cast(
        self,
        interpreter: Interpreter,
        op: UnrealizedConversionCastOp,
        args: tuple[Any, ...],
    ):
        return tuple(
            self.cast_value(o.typ, r.typ, arg)
            for (o, r, arg) in zip(op.operands, op.results, args)
        )
