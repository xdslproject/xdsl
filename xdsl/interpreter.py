from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Any, Callable, Generator, Iterable, TypeAlias, TypeVar, ParamSpec

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OperationInvT, SSAValue, Operation
from xdsl.utils.exceptions import InterpretationError


@dataclass
class InterpreterFunctions:
    """
    Holds the Python implementations for Operations. Users should
    subclass this class, and define the functions to run during interpretation.
    For example:

    ``` python
    @register_impls
    class ArithFunctions(InterpreterFunctions):

        @impl(arith.Addi)
        def run_addi(self, interpreter: Interpreter, op: arith.Addi,
                     args: tuple[Any, ...]) -> tuple[Any, ...]:
            lhs, rhs = args
            return lhs + rhs,
    ```

    The interpreter will take care of fetching the Python values associated
    with the operand SSAValues, and setting the return values to the
    appropriate OpResults.

    To override the definition of an operation implementation, subclass the
    class to override, and redefine the functions, annotating them with
    `@impl`.

    ``` python
    @register_impls
    class DebugArithFunctions(ArithFunctions):

        @impl(arith.Addi)
        def run_addi(self, interpreter: Interpreter, op: arith.Addi,
                     args: tuple[Any, ...]) -> tuple[Any, ...]:
            lhs, rhs = args
            print(lhs, rhs, lhs + rhs)
            return lhs + rhs,
    ```
    """

    @classmethod
    def _impls(
        cls,
    ) -> Iterable[tuple[type[Operation], OpImpl[InterpreterFunctions, Operation]]]:
        try:
            impl_dict = getattr(cls, _IMPL_DICT)
            return impl_dict.items()
        except AttributeError as e:
            raise ValueError(f"Use `@register_impls` on class {cls.__name__}") from e


_FT = TypeVar("_FT", bound=InterpreterFunctions)

_IMPL_OP_TYPE = "__impl_op_type"
_IMPL_DICT = "__impl_dict"

P = ParamSpec("P")


def impl(
    op_type: type[OperationInvT],
) -> Callable[[OpImpl[_FT, OperationInvT]], OpImpl[_FT, OperationInvT]]:
    """
    Marks the Python implementation of an xDSL `Operation` instance, to be used
    by an `Interpreter`. The Interpreter will fetch the Python values
    associated with the operands from the current environment, and pass them as
    the `args` parameter. The returned values are assigned to the `results`
    values.

    See `InterpreterFunctions`
    """

    def annot(func: OpImpl[_FT, OperationInvT]) -> OpImpl[_FT, OperationInvT]:
        setattr(func, _IMPL_OP_TYPE, op_type)
        return func

    return annot


def register_impls(ft: type[_FT]) -> type[_FT]:
    """
    Enumerates the methods on a given class, and registers the ones marked with
    `@impl` in a way that an `Interpreter` instance can find them for dynamic
    dispatch during interpretation.

    See `InterpreterFunctions`
    """
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

    return ft


@dataclass
class _InterpreterFunctionImpls:
    """
    Used to combine multiple function implementations. The operation
    implementations need to be passed the instance of the Functions class,
    so we keep a `(Functions, OpImpl)` tuple for every Operation type.
    """

    _impl_dict: dict[
        type[Operation],
        tuple[InterpreterFunctions, OpImpl[InterpreterFunctions, Operation]],
    ] = field(default_factory=dict)

    def register_from(self, ft: InterpreterFunctions, /, override: bool):
        impls = ft._impls()  # pyright: ignore[reportPrivateUsage]
        for op_type, impl in impls:
            if op_type in self._impl_dict and not override:
                raise ValueError(
                    "Attempting to register implementation for op of type "
                    f"{op_type}, but type already registered"
                )

            self._impl_dict[op_type] = (ft, impl)

    def run(
        self, interpreter: Interpreter, op: Operation, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        if type(op) not in self._impl_dict:
            raise InterpretationError(
                f"Could not find interpretation function for op {op.name}"
            )
        ft, impl = self._impl_dict[type(op)]
        return impl(ft, interpreter, op, args)


@dataclass
class InterpreterContext:
    """
    Class holding the Python values associated with SSAValues during an
    interpretation context. An environment is a stack of scopes, values are
    assigned to the current scope, but can be fetched from a parent scope.
    """

    name: str = field(default="unknown")
    parent: InterpreterContext | None = None
    env: dict[SSAValue, Any] = field(default_factory=dict)

    def __getitem__(self, key: SSAValue) -> Any:
        """
        Fetch key from environment. Attempts to first fetch from current scope,
        then from parent scopes. Raises Interpretation error if not found.
        """
        if key in self.env:
            return self.env[key]
        if self.parent is not None:
            return self.parent[key]
        raise InterpretationError(f"Could not find value for {key} in {self}")

    def __setitem__(self, key: SSAValue, value: Any):
        """
        Assign key to current scope. Raises InterpretationError if key already
        assigned to.
        """
        if key in self.env:
            raise InterpretationError(
                f"Attempting to register SSAValue {value} for name {key}"
                f", but value with that name already exists in {self}"
            )
        self.env[key] = value

    def stack(self) -> Generator[InterpreterContext, None, None]:
        """
        Iterates through scopes starting with the root scope.
        """
        if self.parent is not None:
            yield from self.parent.stack()
        yield self

    def __format__(self, __format_spec: str) -> str:
        return "/".join(c.name for c in self.stack())


@dataclass
class Interpreter:
    """
    An extensible interpreter, initialised with a Module to interpret. The
    implementation for each Operation subclass should be provided via a
    `InterpretationFunctions` instance. Interpretations can be overridden, and
    the override must be specified explicitly, by passing `override=True` to
    the `register_functions` method.
    """

    module: ModuleOp
    _impls: _InterpreterFunctionImpls = field(default_factory=_InterpreterFunctionImpls)
    _ctx: InterpreterContext = field(
        default_factory=lambda: InterpreterContext(name="root")
    )
    file: IO[str] | None = field(default=None)

    def get_values(self, values: Iterable[SSAValue]) -> tuple[Any, ...]:
        """
        Get values from current environment.
        """
        return tuple(self._ctx[value] for value in values)

    def set_values(self, pairs: Iterable[tuple[SSAValue, Any]]):
        """
        Set values to current scope.
        Raises InterpretationError if len(ssa_values) != len(result_values), or
        if SSA value already has a Python value in the current scope.
        """
        for ssa_value, result_value in pairs:
            self._ctx[ssa_value] = result_value

    def push_scope(self, name: str = "unknown") -> None:
        """
        Create new scope in current environment, with optional custom `name`.
        """
        self._ctx = InterpreterContext(name, self._ctx)

    def pop_scope(self) -> None:
        """
        Discard the current scope, and all the values registered in it. Sets
        parent scope of current scope to new current scope.
        Raises InterpretationError if current scope is root scope.
        """
        if self._ctx.parent is None:
            raise InterpretationError("Attempting to pop root env")

        self._ctx = self._ctx.parent

    def register_implementations(
        self, impls: InterpreterFunctions, /, override: bool = False
    ) -> None:
        """
        Register implementations for operations defined in given
        `InterpreterFunctions` object. Raise InterpretationError if an
        operation already has an implementation registered, unless override is
        set to True.
        """
        self._impls.register_from(impls, override=override)

    def run(self, op: Operation):
        """
        Fetches the implemetation for the given op, passes it the Python values
        associated with the SSA operands, and assigns the results to the
        operation's results.
        """
        inputs = self.get_values(op.operands)
        results = self._impls.run(self, op, inputs)
        self.interpreter_assert(
            len(op.results) == len(results), "Incorrect number of results"
        )
        self.set_values(zip(op.results, results))

    def run_module(self):
        """Starts execution of `self.module`"""
        self.run(self.module)

    def print(self, *args: Any, **kwargs: Any):
        """Print to current file."""
        print(*args, **kwargs, file=self.file)

    def interpreter_assert(self, condition: bool, message: str | None = None):
        """Raise InterpretationError if condition is not satisfied."""
        if not condition:
            raise InterpretationError(f"AssertionError: ({self._ctx})({message})")


OpImpl: TypeAlias = Callable[
    [_FT, Interpreter, OperationInvT, tuple[Any, ...]], tuple[Any, ...]
]

_ImplDict: TypeAlias = dict[type[Operation], OpImpl[InterpreterFunctions, Operation]]
