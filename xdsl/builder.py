from __future__ import annotations

from typing import ParamSpec, Callable, TypeVar, Concatenate

from dataclasses import dataclass, field

from xdsl.ir import Operation, OperationInvT, Attribute, Region, Block
from xdsl.dialects.builtin import FunctionType

_P = ParamSpec('_P')
_T = TypeVar('_T')


@dataclass
class Builder:
    _ops: list[Operation] = field(default_factory=list)

    def _get_ops(self) -> list[Operation]:
        return self._ops

    def add_op(self, op: Operation):
        self._ops.append(op)

    def create(self, func: Callable[_P, OperationInvT], *args: _P.args,
               **kwargs: _P.kwargs) -> OperationInvT:
        op = func(*args, **kwargs)
        self.add_op(op)
        return op

    @staticmethod
    def region(func: Callable[[Builder], None]) -> Region:

        builder = Builder()

        func(builder)

        ops = builder._get_ops()
        return Region.from_operation_list(ops)

    @staticmethod
    def callable_region(
        input_types: list[Attribute], return_types: list[Attribute]
    ) -> Callable[[Callable[Concatenate[Builder, _P], None]], tuple[
            Region, FunctionType]]:

        def wrapper(
            func: Callable[Concatenate[Builder, _P], None]
        ) -> tuple[Region, FunctionType]:

            def impl(*args: _P.args, **kwargs: _P.kwargs) -> list[Operation]:
                builder = Builder()

                func(builder, *args, **kwargs)

                return builder._get_ops()

            region = Region.from_block_list(
                [Block.from_callable(input_types, impl)])
            ftype = FunctionType.from_lists(input_types, return_types)
            return region, ftype

        return wrapper
