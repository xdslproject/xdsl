from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Any, Callable, Generator, Iterable, Sequence, TypeAlias

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OperationInvT, SSAValue, Operation
from xdsl.utils.exceptions import IntepretationError


@dataclass
class InterpreterFunctionTable:
    _impl_by_op_type: dict[type[Operation],
                           OpImpl[Operation]] = field(default_factory=dict)

    def register_op(self, op_type: type[OperationInvT],
                    func: OpImpl[OperationInvT]):
        """
        Registers a Python function to run for a given Operation type.
        If the type already exists, will raise a ValueError.
        """
        if op_type in self._impl_by_op_type:
            raise ValueError(
                f"Registering func for Operation type {op_type}, already registered. "
                "Pass `override=True` if you would like to override the existing definition."
            )
        self._impl_by_op_type[op_type] = func  # type: ignore

    def __contains__(self, op_type: type[Operation]) -> bool:
        return op_type in self._impl_by_op_type

    def register(
        self,
        op_type: type[OperationInvT],
        /,
        override: bool = False
    ) -> Callable[[OpImpl[OperationInvT]], OpImpl[OperationInvT]]:
        """
        Registers a Python function to run for a given Operation type.
        If the type already exists, will raise a ValueError.
        """

        def wrapper(func: OpImpl[OperationInvT]):
            self.register_op(op_type, func)
            return func

        return wrapper

    def run(self, interpreter: Intepreter, op: Operation,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
        return self._impl_by_op_type[type(op)](interpreter, op, args)


@dataclass
class CompoundInterpretationFunctionTable(InterpreterFunctionTable):

    def register_from(self, other: InterpreterFunctionTable, /,
                      override: bool):
        """Register each operation in other, one by one."""
        for op_type, func in other._impl_by_op_type.items():
            self.register_op(op_type, func)


@dataclass
class IntepretationEnv:
    """
    Class holding the Python values associated with SSAValues during an interpretation
    context. An environment is a stack of scopes, values are assigned to the current
    scope, but can be fetched from a parent scope.
    """

    name: str = field(default="unknown")
    parent: IntepretationEnv | None = None
    env: dict[SSAValue, Any] = field(default_factory=dict)

    def __getitem__(self, key: SSAValue) -> Any:
        """
        Fetch key from environment. Attempts to first fetch from current scope, then
        from parent scopes. Raises Interpretation error if not found.
        """
        if key in self.env:
            return self.env[key]
        if self.parent is not None:
            return self.parent[key]
        raise IntepretationError(f'Could not find value for {key} in {self}')

    def __setitem__(self, key: SSAValue, value: Any):
        """
        Assign key to current scope. Raises InterpretationError if key already assigned to.
        """
        if key in self.env:
            raise IntepretationError(
                f'Attempting to register SSAValue {value} for name {key}'
                f', but value with that name already exists in {self}')
        self.env[key] = value

    def stack(self) -> Generator[IntepretationEnv, None, None]:
        """
        Iterates through scopes starting with the root scope.
        """
        if self.parent is not None:
            yield from self.parent.stack()
        yield self

    def __format__(self, __format_spec: str) -> str:
        return '/'.join(c.name for c in self.stack())


@dataclass
class Intepreter:
    """
    
    """

    module: ModuleOp
    _function_table: CompoundInterpretationFunctionTable = field(
        default_factory=CompoundInterpretationFunctionTable)
    _env: IntepretationEnv = field(
        default_factory=lambda: IntepretationEnv(name='root'))
    file: IO[str] | None = field(default=None)

    def get_values(self, values: Iterable[SSAValue]) -> tuple[Any, ...]:
        """
        Get values from current environment.
        """
        return tuple(self._env[value] for value in values)

    def set_values(self, ssa_values: Sequence[SSAValue],
                   result_values: Sequence[Any]):
        """
        Set values to current scope.
        Raises InterpretationError if len(ssa_values) != len(result_values), or if
        SSA value already has a Python value in the current scope.
        """
        self._assert(
            len(ssa_values) == len(result_values),
            f'{[f"{ssa_value}" for ssa_value in ssa_values]}, {result_values}')
        for ssa_value, result_value in zip(ssa_values, result_values):
            self._env[ssa_value] = result_value

    def push_scope(self, name: str = 'unknown') -> None:
        """
        Create new scope in current environment, with optional custom `name`
        """
        self._env = IntepretationEnv(name, self._env)

    def pop_scope(self) -> None:
        """
        Discard the current scope, and all the values registered in it. Sets parent scope
        of current scope to new current scope.
        Raises InterpretationError if current scope is root scope.
        """
        if self._env.parent is None:
            raise IntepretationError('Attempting to pop root env')

        self._env = self._env.parent

    def register_functions(self,
                           funcs: InterpreterFunctionTable,
                           /,
                           override: bool = False) -> None:
        """
        Register implementations for operations defined in given `Functions` object.
        Raise InterpretationError if an operation already has an implementation registered
        , unless override is set to True.
        """
        self._function_table.register_from(funcs, override=override)

    def run(self, op: Operation):
        op_type = type(op)
        if op_type not in self._function_table:
            raise IntepretationError(
                f'Could not find intepretation function for op {op.name}')

        inputs = self.get_values(op.operands)
        results = self._function_table.run(self, op, inputs)
        self.set_values(tuple(op.results), results)

    def print(self, *args: Any, **kwargs: Any):
        """Print to current file."""
        print(*args, **kwargs, file=self.file)

    def _assert(self, condition: bool, message: str | None = None):
        "Raise InterpretationError if condition is not satisfied."
        if not condition:
            raise IntepretationError(
                f'AssertionError: ({self._env})({message})')


OpImpl: TypeAlias = Callable[[Intepreter, OperationInvT, tuple[Any, ...]],
                             tuple[Any, ...]]
