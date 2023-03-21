from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Any, Callable, Generator

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OperationInvT, SSAValue, Operation
from xdsl.utils.exceptions import IntepretationError


@dataclass
class Functions:
    functions: dict[type[Operation],
                    Callable[[Intepreter, Operation, tuple[Any, ...]],
                             tuple[Any, ...]]] = field(default_factory=dict)

    def register_op(
            self,
            op_type: type[OperationInvT],
            func: Callable[[Intepreter, OperationInvT, tuple[Any, ...]],
                           tuple[Any, ...]],
            /,
            override: bool = False):
        '''
        Registers a Python function to run for a given Operation type.
        If the type already exists, will raise a ValueError. To override an existing
        implementation, pass `override=True`.
        '''
        if op_type in self.functions and not override:
            raise ValueError(
                f"Registering func for Operation type {op_type}, already registered. "
                "Pass `override=True` if you would like to override the existing definition."
            )
        self.functions[op_type] = func  # type: ignore

    def op_types(self) -> set[type[Operation]]:
        return set(self.functions.keys())

    def register(
        self,
        op_type: type[OperationInvT],
        /,
        override: bool = False
    ) -> Callable[[
            Callable[[Intepreter, OperationInvT, tuple[Any, ...]], tuple[Any,
                                                                         ...]]
    ], Callable[[Intepreter, OperationInvT, tuple[Any, ...]], tuple[Any,
                                                                    ...]]]:

        def wrapper(
            func: Callable[[Intepreter, OperationInvT, tuple[Any, ...]],
                           tuple[Any, ...]]):
            self.register_op(op_type, func, override=override)
            return func

        return wrapper

    def run(self, interpreter: Intepreter, op: Operation,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
        return self.functions[type(op)](interpreter, op, args)

    def register_from(self, other: Functions, override: bool = False):
        '''Register each operation in other, one by one.'''
        for op_type, func in other.functions.items():
            self.register_op(op_type, func, override=override)


@dataclass
class IntepretationEnv:
    name: str = field(default="unknown")
    parent: IntepretationEnv | None = None
    env: dict[SSAValue, Any] = field(default_factory=dict)

    def __getitem__(self, key: SSAValue) -> Any:
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
    file: IO[str] | None = field(default=None)

    def get_values(self, values: tuple[SSAValue, ...]) -> tuple[Any, ...]:
        return tuple(self._env[value] for value in values)

    def set_values(self, ssa_values: tuple[SSAValue, ...],
                   result_values: tuple[Any, ...]):
        self._assert(
            len(ssa_values) == len(result_values),
            f'{[f"{ssa_value}" for ssa_value in ssa_values]}, {result_values}')
        for ssa_value, result_value in zip(ssa_values, result_values):
            self._env[ssa_value] = result_value

    def push_scope(self, name: str = 'unknown') -> None:
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
