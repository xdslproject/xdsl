from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    IO,
    Any,
    Callable,
    Generator,
    Iterable,
    NamedTuple,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, OperationInvT, SSAValue
from xdsl.ir.core import Attribute, Block, Region
from xdsl.traits import CallableOpInterface, IsTerminator, SymbolOpInterface
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

    To register an implementation of a cast for UnrealizedConversionCastOp, use
    `impl_cast`, like so:

    ``` python
    @register_impls
    class ArithFunctions(InterpreterFunctions):
        @impl_cast(IntegerType, IndexType)
        def cast_integer_to_index(
            self,
            input_type: IntegerType,
            output_type: IndexType,
            value: Any,
        ) -> Any:
            # Both input and output represented by a Python `int`
            return value
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

    @classmethod
    def _cast_impls(
        cls,
    ) -> Iterable[
        tuple[
            tuple[type[Attribute], type[Attribute]],
            CastImpl[InterpreterFunctions, Attribute, Attribute],
        ]
    ]:
        try:
            impl_dict = getattr(cls, _CAST_IMPL_DICT)
            return impl_dict.items()
        except AttributeError as e:
            raise ValueError(f"Use `@register_impls` on class {cls.__name__}") from e


_FT = TypeVar("_FT", bound=InterpreterFunctions)

_IMPL_OP_TYPE = "__impl_op_type"
_CAST_IMPL_TYPES = "__cast_impl_types"
_IMPL_DICT = "__impl_dict"
_CAST_IMPL_DICT = "__cast_impl_dict"

P = ParamSpec("P")


def impl(
    op_type: type[OperationInvT],
) -> Callable[[NonTerminatorOpImpl[_FT, OperationInvT]], OpImpl[_FT, OperationInvT]]:
    """
    Marks the Python implementation of an xDSL `Operation` instance, to be used
    by an `Interpreter`. The Interpreter will fetch the Python values
    associated with the operands from the current environment, and pass them as
    the `args` parameter. The returned values are assigned to the `results`
    values.

    See `InterpreterFunctions`
    """

    if op_type.has_trait(IsTerminator):
        raise ValueError(
            "Operations that are terminators must use `impl_terminator` annotation"
        )

    def annot(
        func: NonTerminatorOpImpl[_FT, OperationInvT]
    ) -> OpImpl[_FT, OperationInvT]:
        def impl(
            ft: _FT, interpreter: Interpreter, op: OperationInvT, values: PythonValues
        ) -> OpImplResult:
            return OpImplResult(func(ft, interpreter, op, values), None)

        setattr(impl, _IMPL_OP_TYPE, op_type)
        return impl

    return annot


def impl_terminator(
    op_type: type[OperationInvT],
) -> Callable[[TerminatorOpImpl[_FT, OperationInvT]], OpImpl[_FT, OperationInvT]]:
    """
    Marks the Python implementation of an xDSL `Operation` instance, to be used
    by an `Interpreter`. The Interpreter will fetch the Python values
    associated with the operands from the current environment, and pass them as
    the `args` parameter. The returned values are assigned to the `results`
    values.

    See `InterpreterFunctions`
    """

    if not op_type.has_trait(IsTerminator):
        raise ValueError(
            "Operations that are not terminators must use `impl` annotation"
        )

    def annot(func: TerminatorOpImpl[_FT, OperationInvT]) -> OpImpl[_FT, OperationInvT]:
        def impl(
            ft: _FT, interpreter: Interpreter, op: OperationInvT, values: PythonValues
        ) -> OpImplResult:
            successor, args = func(ft, interpreter, op, values)
            return OpImplResult(args, successor)

        setattr(impl, _IMPL_OP_TYPE, op_type)
        return impl

    return annot


def impl_cast(
    input_type: type[_AttributeInvT0],
    output_type: type[_AttributeInvT1],
) -> Callable[
    [CastImpl[_FT, _AttributeInvT0, _AttributeInvT1]],
    CastImpl[_FT, _AttributeInvT0, _AttributeInvT1],
]:
    """
    Marks the Python implementation of a value cast from one type to another. The
    `cast_value` method on `Interpreter` will call into this implementation for matching
    input and output types.

    See `InterpreterFunctions` for more documentation.
    """

    def annot(
        func: CastImpl[_FT, _AttributeInvT0, _AttributeInvT1]
    ) -> CastImpl[_FT, _AttributeInvT0, _AttributeInvT1]:
        setattr(func, _CAST_IMPL_TYPES, (input_type, output_type))
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
    cast_impl_dict: _CastImplDict = {}

    for cls in ft.mro():
        # Iterate from subclass through superclasses
        # Assign definitions, unless they've been redefined in a subclass
        for val in cls.__dict__.values():
            if _IMPL_OP_TYPE in val.__dir__():
                # This is an annotated operation implementation
                op_type = getattr(val, _IMPL_OP_TYPE)
                if op_type not in impl_dict:
                    # subclass overrides superclass definition
                    impl_dict[op_type] = val
            elif _CAST_IMPL_TYPES in val.__dir__():
                # This is an annotated cast implementation
                types = getattr(val, _CAST_IMPL_TYPES)
                if types not in cast_impl_dict:
                    # subclass overrides superclass definition
                    cast_impl_dict[types] = val

    setattr(ft, _IMPL_DICT, impl_dict)
    setattr(ft, _CAST_IMPL_DICT, cast_impl_dict)

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
    _cast_impl_dict: dict[
        tuple[type[Attribute], type[Attribute]],
        tuple[
            InterpreterFunctions, CastImpl[InterpreterFunctions, Attribute, Attribute]
        ],
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

        cast_impls = ft._cast_impls()  # pyright: ignore[reportPrivateUsage]
        for types, impl in cast_impls:
            if types in self._cast_impl_dict and not override:
                raise ValueError(
                    "Attempting to register implementation for cast with types "
                    f"{types}, but types already registered"
                )

            self._cast_impl_dict[types] = (ft, impl)

    def run(
        self, interpreter: Interpreter, op: Operation, args: tuple[Any, ...]
    ) -> OpImplResult:
        if type(op) not in self._impl_dict:
            raise InterpretationError(
                f"Could not find interpretation function for op {op.name}"
            )
        ft, impl = self._impl_dict[type(op)]
        return impl(ft, interpreter, op, args)

    def cast(
        self,
        input_type: Attribute,
        output_type: Attribute,
        value: Any,
    ) -> Any:
        types = (type(input_type), type(output_type))
        if types not in self._cast_impl_dict:
            raise InterpretationError(
                f"Could not find cast implementation for types {input_type}, {output_type}"
            )
        ft, impl = self._cast_impl_dict[types]
        return impl(ft, input_type, output_type, value)


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
    _symbol_table: dict[str, Operation] | None = None

    @property
    def symbol_table(self) -> dict[str, Operation]:
        if self._symbol_table is None:
            self._symbol_table = {}

            for op in self.module.walk():
                if op.has_trait(SymbolOpInterface):
                    symbol = SymbolOpInterface.get_sym_attr_name(op)
                    self._symbol_table[symbol.data] = op
        return self._symbol_table

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

    def run_op(self, op: Operation | str, inputs: PythonValues) -> PythonValues:
        """
        Calls the implementation for the given operation.
        """
        if isinstance(op, str):
            op = self.get_op_for_symbol(op)

        result = self._impls.run(self, op, inputs)
        return result.values

    def call_op(self, op: Operation | str, inputs: PythonValues) -> PythonValues:
        """
        Calls the implementation for the given operation.
        """
        if isinstance(op, str):
            name = op
            op = self.get_op_for_symbol(op)
        else:
            name = "unknown"

        interface = op.get_trait(CallableOpInterface)

        self.interpreter_assert(
            interface is not None,
            f"Operation {op.name} does not have trait CallableOpInterface",
        )
        assert interface is not None

        body = interface.get_callable_region(op)

        results = self.run_ssacfg_region(body, inputs, name)
        assert results is not None
        return results

    def run_ssacfg_region(
        self, region: Region, args: PythonValues, name: str = "unknown"
    ) -> PythonValues:
        """
        Interpret an SSACFG-semantic Region.
        Creates a new scope, then executes the first block in the region. The first block
        is expected to return the results of the region directly.
        """
        results = ()
        if not region.blocks:
            return results

        scope_count = 0
        block = region.blocks[0]

        while block is not None:
            self.push_scope(name)
            scope_count += 1
            self.set_values(zip(block.args, args))

            op: Operation | None = block.first_op
            block = None

            while op is not None:
                inputs = self.get_values(op.operands)
                result = self._impls.run(self, op, inputs)
                self.interpreter_assert(
                    len(op.results) == len(result.values),
                    f"Incorrect number of results for op {op.name}",
                )
                self.set_values(zip(op.results, result.values))

                if result.terminator_value is not None:
                    match result.terminator_value:
                        case ReturnedValues():
                            # update results and break out of outer loop
                            results = result.terminator_value.values
                            break
                        case Successor():
                            # block won't be None, so only break out of inner loop
                            block, args = result.terminator_value
                            break

                # Set up next iteration
                op = op.next_op

        # Pop as many scopes as we entered blocks
        for _ in range(scope_count):
            self.pop_scope()
        return results

    def cast_value(self, o: Attribute, r: Attribute, value: Any) -> Any:
        """
        If the type of the operand and result are not the same, then look up the
        user-provided conversion function.
        """
        if o == r:
            return value

        return self._impls.cast(o, r, value)

    def get_op_for_symbol(self, symbol: str) -> Operation:
        if symbol in self.symbol_table:
            return self.symbol_table[symbol]
        else:
            raise InterpretationError(f'Could not find symbol "{symbol}"')

    def print(self, *args: Any, **kwargs: Any):
        """Print to current file."""
        print(*args, **kwargs, file=self.file)

    def interpreter_assert(self, condition: bool, message: str | None = None):
        """Raise InterpretationError if condition is not satisfied."""
        if not condition:
            raise InterpretationError(f"AssertionError: ({self._ctx})({message})")


PythonValues: TypeAlias = tuple[Any, ...]


class ReturnedValues(NamedTuple):
    values: PythonValues


class Successor(NamedTuple):
    block: Block
    args: PythonValues


TerminatorValue: TypeAlias = ReturnedValues | Successor


class OpImplResult(NamedTuple):
    """
    The result of interpreting an Operation. If and only if the Operation is a terminator,
    it must set the terminator_value.
    """

    values: PythonValues
    terminator_value: TerminatorValue | None


NonTerminatorOpImpl: TypeAlias = Callable[
    [_FT, Interpreter, OperationInvT, PythonValues], PythonValues
]

TerminatorOpImpl: TypeAlias = Callable[
    [_FT, Interpreter, OperationInvT, PythonValues],
    tuple[TerminatorValue, PythonValues],
]

OpImpl: TypeAlias = Callable[
    [_FT, Interpreter, OperationInvT, PythonValues], OpImplResult
]

_AttributeInvT0 = TypeVar("_AttributeInvT0", bound=Attribute)
_AttributeInvT1 = TypeVar("_AttributeInvT1", bound=Attribute)
CastImpl: TypeAlias = Callable[
    [_FT, _AttributeInvT0, _AttributeInvT1, Any],
    Any,
]

_ImplDict: TypeAlias = dict[type[Operation], OpImpl[InterpreterFunctions, Operation]]

_CastImplDict: TypeAlias = dict[
    tuple[type[Attribute], type[Attribute]],
    CastImpl[InterpreterFunctions, Attribute, Attribute],
]
