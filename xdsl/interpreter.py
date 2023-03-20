from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Callable, Generator, TypeVar
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue, Operation

_OperationInvT = TypeVar('_OperationInvT', bound=Operation)


class IntepretationError(Exception):
    pass


@dataclass
class Functions:
    functions: dict[type[Operation],
                    Callable[[Intepreter, Operation, tuple[Any, ...]],
                             tuple[Any, ...]]] = field(default_factory=dict)

    def register_op(
        self, op_type: type[_OperationInvT],
        func: Callable[[Intepreter, _OperationInvT, tuple[Any, ...]],
                       tuple[Any, ...]]):
        self.functions[op_type] = func  # type: ignore

    def op_types(self) -> set[type[Operation]]:
        return set(self.functions.keys())

    def register(
        self, op_type: type[_OperationInvT]
    ) -> Callable[[
            Callable[[Intepreter, _OperationInvT, tuple[Any, ...]], tuple[Any,
                                                                          ...]]
    ], Callable[[Intepreter, _OperationInvT, tuple[Any, ...]], tuple[Any,
                                                                     ...]]]:

        def wrapper(
            func: Callable[[Intepreter, _OperationInvT, tuple[Any, ...]],
                           tuple[Any, ...]]):
            self.register_op(op_type, func)
            return func

        return wrapper

    def run(self, interpreter: Intepreter, op: Operation,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.functions[type(op)](interpreter, op, args)

    def register_from(self, other: Functions):
        '''If there are duplicate definitions, the `other` will override `self`'''
        self.functions.update(other.functions)


@dataclass
class IntepretationEnv:
    name: str = field(default="unknown")
    parent: IntepretationEnv | None = None
    env: dict[SSAValue, Any] = field(default_factory=dict)

    def __getitem__(self, key: SSAValue) -> SSAValue:
        if key in self.env:
            return self.env[key]
        if self.parent is not None:
            return self.parent[key]
        raise IntepretationError(f'Could not find value for {key}')

    def __setitem__(self, key: SSAValue, value: Any):
        if key in self.env:
            raise IntepretationError(
                f'Attempting to register SSAValue {value} for name {key}'
                ', but value with that name already exists')
        self.env[key] = value

    def stack(self) -> Generator[IntepretationEnv, None, None]:
        if self.parent is not None:
            yield from self.parent.stack()
        yield self

    def stack_description(self) -> str:
        return '/'.join(c.name for c in self.stack())


@dataclass
class Intepreter:

    module: ModuleOp
    _function_table: Functions = field(default_factory=Functions)
    _env: IntepretationEnv = field(
        default_factory=lambda: IntepretationEnv(name='root'))
    file: StringIO | None = field(default=None)

    def get_values(self, values: tuple[SSAValue, ...]) -> tuple[Any, ...]:
        return tuple(self._env[value] for value in values)

    def set_values(self, ssa_values: tuple[SSAValue, ...],
                   result_values: tuple[Any, ...]):
        self._assert(
            len(ssa_values) == len(result_values),
            f'{[f"{ssa_value}" for ssa_value in ssa_values]}, {result_values}')
        for ssa_value, result_value in zip(ssa_values, result_values):
            self._env[ssa_value] = result_value

    def push_scope(self, name: str = 'child') -> None:
        self._env = IntepretationEnv(name, self._env)

    def pop_scope(self) -> None:
        if self._env.parent is None:
            raise IntepretationError('Attempting to pop root env')

        self._env = self._env.parent

    def register_functions(self, funcs: Functions) -> None:
        self._function_table.register_from(funcs)

    def run(self, op: Operation):
        op_type = type(op)
        if op_type not in self._function_table.functions:
            raise IntepretationError(
                f'Could not find intepretation function for op {op.name}')

        inputs = self.get_values(op.operands)
        results = self._function_table.run(self, op, inputs)
        self.set_values(tuple(op.results), results)

    def print(self, *args: Any, **kwargs: Any):
        print(*args, **kwargs, file=self.file)

    def _assert(self, condition: bool, message: str | None = None):
        assert condition, f'({self._env.stack_description()})({message})'
