from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Callable, Generator, TypeVar
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue, Operation

_OperationInvT = TypeVar('_OperationInvT', bound=Operation)


class EmulationError(Exception):
    pass


@dataclass
class FunctionTable:
    functions: dict[type[Operation],
                    Callable[[Emulator, Operation, tuple[Any, ...]],
                             tuple[Any, ...]]] = field(default_factory=dict)

    def register_op(self, op_type: type[_OperationInvT],
                    func: Callable[[Emulator, _OperationInvT, tuple[Any, ...]],
                                   tuple[Any, ...]]):
        self.functions[op_type] = func  # type: ignore

    def op_types(self) -> set[type[Operation]]:
        return set(self.functions.keys())

    def register(
        self, op_type: type[_OperationInvT]
    ) -> Callable[[
            Callable[[Emulator, _OperationInvT, tuple[Any, ...]], tuple[Any,
                                                                        ...]]
    ], Callable[[Emulator, _OperationInvT, tuple[Any, ...]], tuple[Any, ...]]]:

        def wrapper(func: Callable[[Emulator, _OperationInvT, tuple[Any, ...]],
                                   tuple[Any, ...]]):
            self.register_op(op_type, func)
            return func

        return wrapper

    def run(self, emulator: Emulator, op: Operation,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.functions[type(op)](emulator, op, args)

    def register_from(self, other: FunctionTable):
        '''If there are duplicate definitions, the `other` will override `self`'''
        self.functions.update(other.functions)


@dataclass
class EmulationContext:
    name: str = field(default="unknown")
    parent: EmulationContext | None = None
    env: dict[SSAValue, Any] = field(default_factory=dict)

    def __getitem__(self, key: SSAValue) -> SSAValue:
        if key in self.env:
            return self.env[key]
        if self.parent is not None:
            return self.parent[key]
        raise EmulationError(f'Could not find value for {key}')

    def __setitem__(self, key: SSAValue, value: Any):
        if key in self.env:
            raise EmulationError(
                f'Attempting to register SSAValue {value} for name {key}'
                ', but value with that name already exists')
        self.env[key] = value

    def stack(self) -> Generator[EmulationContext, None, None]:
        if self.parent is not None:
            yield from self.parent.stack()
        yield self

    def stack_description(self) -> str:
        return '/'.join(c.name for c in self.stack())


@dataclass
class Emulator:

    module: ModuleOp
    _function_table: FunctionTable = field(default_factory=FunctionTable)
    _context: EmulationContext = field(
        default_factory=lambda: EmulationContext(name='root'))
    file: StringIO | None = field(default=None)

    def get_values(self, values: tuple[SSAValue, ...]) -> tuple[Any, ...]:
        return tuple(self._context[value] for value in values)

    def set_values(self, ssa_values: tuple[SSAValue, ...],
                   result_values: tuple[Any, ...]):
        self._assert(
            len(ssa_values) == len(result_values),
            f'{[f"{ssa_value}" for ssa_value in ssa_values]}, {result_values}')
        for ssa_value, result_value in zip(ssa_values, result_values):
            self._context[ssa_value] = result_value

    def push_context(self, name: str = 'child') -> None:
        self._context = EmulationContext(name, self._context)

    def pop_context(self) -> None:
        if self._context.parent is None:
            raise EmulationError('Attempting to pop root env')

        self._context = self._context.parent

    def register_functions(self, funcs: FunctionTable) -> None:
        self._function_table.register_from(funcs)

    def run(self, op: Operation):
        op_type = type(op)
        if op_type not in self._function_table.functions:
            raise EmulationError(
                f'Could not find OperationEmulator for op {op.name}')

        inputs = self.get_values(op.operands)
        results = self._function_table.run(self, op, inputs)
        self.set_values(tuple(op.results), results)

    def print(self, *args: Any, **kwargs: Any):
        print(*args, **kwargs, file=self.file)

    def _assert(self, condition: bool, message: str | None = None):
        assert condition, f'({self._context.stack_description()})({message})'
