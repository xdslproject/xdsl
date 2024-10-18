from collections.abc import Callable
from typing import ClassVar

from jax._src.core import JaxprEqn, Primitive
from jax._src.lax import lax

from xdsl.dialects.stablehlo import AddOp
from xdsl.ir import Operation, SSAValue


class PrimitiveBuilder:
    BUILDER_BY_PRIMITIVE: ClassVar[
        dict[Primitive, Callable[[JaxprEqn, tuple[SSAValue, ...]], Operation]]
    ] = {}

    def __init__(self) -> None:
        super().__init_subclass__()
        # register jaxpr primitives
        self.BUILDER_BY_PRIMITIVE[lax.add_p] = lambda _, args: AddOp(args[0], args[1])  # pyright: ignore[reportUnknownMemberType]
        print(f"Registered in {self.__class__}: {self.BUILDER_BY_PRIMITIVE}")

    @classmethod
    def build(cls, eqn: JaxprEqn, args: tuple[SSAValue, ...]) -> Operation:
        primitive = eqn.primitive
        if primitive not in cls.BUILDER_BY_PRIMITIVE:
            raise ValueError(f"No builder for primitive: {primitive}")

        builder = cls.BUILDER_BY_PRIMITIVE[primitive]
        res = builder(eqn, args)
        return res
