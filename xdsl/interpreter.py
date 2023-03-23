from __future__ import annotations

from dataclasses import dataclass, field
from typing import (IO, Any, Callable, Generator, Iterable, Sequence,
                    TypeAlias, TypeVar, ParamSpec)

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OperationInvT, SSAValue, Operation
from xdsl.utils.exceptions import InterpretationError


@dataclass
class InterpreterFunctionTable:

    @classmethod
    def impls(
        cls
    ) -> Iterable[tuple[type[Operation], OpImpl[InterpreterFunctionTable,
                                                Operation]]]:
        ...


_FT = TypeVar('_FT', bound=InterpreterFunctionTable)

_IMPL_OP_TYPE = '__impl_op_type'
_IMPL_DICT = '__impl_dict'

P = ParamSpec('P')


def impl(
    op_type: type[OperationInvT]
) -> Callable[[OpImpl[_FT, OperationInvT]], OpImpl[_FT, OperationInvT]]:

    def annot(func: OpImpl[_FT, OperationInvT]) -> OpImpl[_FT, OperationInvT]:
        setattr(func, _IMPL_OP_TYPE, op_type)
        return func

    return annot


def function_table(ft: type[_FT]) -> type[_FT]:
    impl_dict: _ImplDict = {}
    for cls in ft.mro():
        # Iterate from subclass through superclasses
        # Assign definitions, unless they've been redefined in a subclass
        for val in cls.__dict__.values():
            if _IMPL_OP_TYPE in val.__dir__():
                # This is an annotated function
                op_type = getattr(val, _IMPL_OP_TYPE)
                if op_type not in impl_dict:
                    # subclass overrides superclass definition
                    impl_dict[op_type] = val  # type: ignore
    setattr(ft, _IMPL_DICT, impl_dict)

    @classmethod
    def impls(
        cls: type[_FT]
    ) -> Iterable[tuple[type[Operation], OpImpl[InterpreterFunctionTable,
                                                Operation]]]:
        return impl_dict.items()

    setattr(ft, 'impls', impls)
    return ft


@dataclass
class CompoundInterpretationFunctionTable(InterpreterFunctionTable):

    _impl_dict: dict[type[Operation],
                     tuple[InterpreterFunctionTable,
                           OpImpl[InterpreterFunctionTable,
                                  Operation]]] = field(default_factory=dict)

    def register_from(self, ft: InterpreterFunctionTable, /, override: bool):
        for op_type, impl in ft.impls():
            if op_type in self._impl_dict and not override:
                raise ValueError(
                    "Attempting to register implementation for op of type "
                    f"{op_type}, but type already registered")

            self._impl_dict[op_type] = (ft, impl)

    def run(self, interpreter: Interpreter, op: Operation,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
        if type(op) not in self._impl_dict:
            raise InterpretationError(
                f'Could not find interpretation function for op {op.name}')
        ft, impl = self._impl_dict[type(op)]
        return impl(ft, interpreter, op, args)


@dataclass
class InterpretationEnv:
    """
    Class holding the Python values associated with SSAValues during an interpretation
    context. An environment is a stack of scopes, values are assigned to the current
    scope, but can be fetched from a parent scope.
    """

    name: str = field(default="unknown")
    parent: InterpretationEnv | None = None
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
        raise InterpretationError(f'Could not find value for {key} in {self}')

    def __setitem__(self, key: SSAValue, value: Any):
        """
        Assign key to current scope. Raises InterpretationError if key already assigned to.
        """
        if key in self.env:
            raise InterpretationError(
                f'Attempting to register SSAValue {value} for name {key}'
                f', but value with that name already exists in {self}')
        self.env[key] = value

    def stack(self) -> Generator[InterpretationEnv, None, None]:
        """
        Iterates through scopes starting with the root scope.
        """
        if self.parent is not None:
            yield from self.parent.stack()
        yield self

    def __format__(self, __format_spec: str) -> str:
        return '/'.join(c.name for c in self.stack())


@dataclass
class Interpreter:
    """
    An extensible interpreter, initialised with a Module to intperpret. The implementation
    for each Operation subclass should be provided via a `InterpretationFunctionTable`
    instance. Interpretations can be overridden, and the override must be specified
    explicitly, by passing `override=True` to the `register_functions` method.
    """

    module: ModuleOp
    _function_table: CompoundInterpretationFunctionTable = field(
        default_factory=CompoundInterpretationFunctionTable)
    _env: InterpretationEnv = field(
        default_factory=lambda: InterpretationEnv(name='root'))
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
        self._env = InterpretationEnv(name, self._env)

    def pop_scope(self) -> None:
        """
        Discard the current scope, and all the values registered in it. Sets parent scope
        of current scope to new current scope.
        Raises InterpretationError if current scope is root scope.
        """
        if self._env.parent is None:
            raise InterpretationError('Attempting to pop root env')

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
        """
        Fetches the implemetation for the given op, passes it the Python values
        associated with the SSA operands, and assigns the results to the operation's
        results.
        """
        inputs = self.get_values(op.operands)
        results = self._function_table.run(self, op, inputs)
        self.set_values(tuple(op.results), results)

    def run_module(self):
        """
        Starts execution of `self.module`
        """
        self.run(self.module)

    def print(self, *args: Any, **kwargs: Any):
        """Print to current file."""
        print(*args, **kwargs, file=self.file)

    def _assert(self, condition: bool, message: str | None = None):
        "Raise InterpretationError if condition is not satisfied."
        if not condition:
            raise InterpretationError(
                f'AssertionError: ({self._env})({message})')


OpImpl: TypeAlias = Callable[
    [_FT, Interpreter, OperationInvT, tuple[Any, ...]], tuple[Any, ...]]

_ImplDict: TypeAlias = dict[type[Operation], OpImpl[InterpreterFunctionTable,
                                                    Operation]]
