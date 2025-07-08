from __future__ import annotations

import platform
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import (
    IO,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    ParamSpec,
    TypeAlias,
)

from typing_extensions import TypeVar

from xdsl.dialects.builtin import ModuleOp, SymbolRefAttr
from xdsl.ir import (
    Attribute,
    AttributeInvT,
    Block,
    Operation,
    OperationInvT,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.traits import (
    CallableOpInterface,
    IsTerminator,
    SymbolTable,
)
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.scoped_dict import ScopedDict

PythonValues: TypeAlias = tuple[Any, ...]
"""
A tuple of result values corresponding to the results of the operation being
interpreted.
"""


class ReturnedValues(NamedTuple):
    """
    If the terminator exits the region being interpreted, such as for function returns,
    return these values to yield control back to the parent operation of the block.
    """

    values: PythonValues


class Successor(NamedTuple):
    """
    If the terminator jumps to another block within the region being interpreted, such
    as for jumps, return a block along with the values to pass as arguments.
    """

    block: Block
    args: PythonValues


_FT = TypeVar("_FT", bound="InterpreterFunctions")

TerminatorValue: TypeAlias = ReturnedValues | Successor
"""
A terminator operation either yields control to the parent operation or jumps to a
successor.
"""


class OpImplResult(NamedTuple):
    """
    The result of interpreting an Operation. If and only if the Operation is a terminator,
    it must set the terminator_value.
    """

    values: PythonValues
    terminator_value: TerminatorValue | None


NonTerminatorOpImpl: TypeAlias = Callable[
    [_FT, "Interpreter", OperationInvT, PythonValues], PythonValues
]

TerminatorOpImpl: TypeAlias = Callable[
    [_FT, "Interpreter", OperationInvT, PythonValues],
    tuple[TerminatorValue, PythonValues],
]

OpImpl: TypeAlias = Callable[
    [_FT, "Interpreter", OperationInvT, PythonValues], OpImplResult
]


# region : TypeAliases

_AttributeInvT0 = TypeVar("_AttributeInvT0", bound=Attribute)
_AttributeInvT1 = TypeVar("_AttributeInvT1", bound=Attribute)
CastImpl: TypeAlias = Callable[
    [_FT, _AttributeInvT0, _AttributeInvT1, Any],
    Any,
]
AttrImpl: TypeAlias = Callable[
    [_FT, "Interpreter", Attribute, AttributeInvT],
    Any,
]

_ImplDict: TypeAlias = dict[type[Operation], OpImpl["InterpreterFunctions", Operation]]
_IMPL_DICT: TypeAlias = dict[
    type[Operation],
    tuple["InterpreterFunctions", OpImpl["InterpreterFunctions", Operation]],
]

_CastImplDict: TypeAlias = dict[
    tuple[type[Attribute], type[Attribute]],
    CastImpl["InterpreterFunctions", Attribute, Attribute],
]
_CAST_IMPL_DICT: TypeAlias = dict[
    tuple[type[Attribute], type[Attribute]],
    tuple[
        "InterpreterFunctions", CastImpl["InterpreterFunctions", Attribute, Attribute]
    ],
]

_AttrImplDict: TypeAlias = dict[
    type[Attribute],
    AttrImpl["InterpreterFunctions", TypeAttribute],
]
_ATTR_IMPL_DICT: TypeAlias = dict[
    type[Attribute],
    tuple["InterpreterFunctions", AttrImpl["InterpreterFunctions", Attribute]],
]

ExtFuncImpl: TypeAlias = Callable[
    [_FT, "Interpreter", Operation, PythonValues],
    PythonValues,
]

_ExtFuncImplDict: TypeAlias = dict[
    str,
    ExtFuncImpl["InterpreterFunctions"],
]
_EXT_FUNC_IMPL_DICT: TypeAlias = dict[
    str, tuple["InterpreterFunctions", ExtFuncImpl["InterpreterFunctions"]]
]

_CallableImplDict: TypeAlias = dict[
    type[Operation], NonTerminatorOpImpl["InterpreterFunctions", Operation]
]
_CALLABLE_IMPL_DICT: TypeAlias = dict[
    type[Operation],
    tuple[
        "InterpreterFunctions", NonTerminatorOpImpl["InterpreterFunctions", Operation]
    ],
]
_IMPL_DATA: TypeAlias = dict[type["InterpreterFunctions"], dict[str, Any]]

# endregion

_IMPL_OP_TYPE_KEY = "__impl_op_type"
_CAST_IMPL_TYPES_KEY = "__cast_impl_types"
_ATTR_IMPL_TYPES_KEY = "__attr_impl_types"
_EXT_FUNC_NAME_KEY = "__external_func_name"
_CALLABLE_OP_TYPE_KEY = "__callable_op_type"
_IMPL_DICT_KEY = "__impl_dict"
_CAST_IMPL_DICT_KEY = "__cast_impl_dict"
_ATTR_IMPL_DICT_KEY = "__attr_impl_dict"
_EXT_FUNC_DICT_KEY = "__external_func_dict"
_CALLABLE_IMPL_DICT_KEY = "__callable_impl_dict"


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
            impl_dict = getattr(cls, _IMPL_DICT_KEY)
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
            impl_dict = getattr(cls, _CAST_IMPL_DICT_KEY)
            return impl_dict.items()
        except AttributeError as e:
            raise ValueError(f"Use `@register_impls` on class {cls.__name__}") from e

    @classmethod
    def _attr_impls(
        cls,
    ) -> Iterable[
        tuple[
            type[Attribute],
            AttrImpl[InterpreterFunctions, Attribute],
        ]
    ]:
        try:
            impl_dict = getattr(cls, _ATTR_IMPL_DICT_KEY)
            return impl_dict.items()
        except AttributeError as e:
            raise ValueError(f"Use `@register_impls` on class {cls.__name__}") from e

    @classmethod
    def _ext_impls(
        cls,
    ) -> Iterable[tuple[str, ExtFuncImpl[InterpreterFunctions]]]:
        try:
            impl_dict = getattr(cls, _EXT_FUNC_DICT_KEY)
            return impl_dict.items()
        except AttributeError as e:
            raise ValueError(f"Use `@register_impls` on class {cls.__name__}") from e

    @classmethod
    def _callable_impls(
        cls,
    ) -> Iterable[tuple[type[Operation], OpImpl[InterpreterFunctions, Operation]]]:
        try:
            impl_dict = getattr(cls, _CALLABLE_IMPL_DICT_KEY)
            return impl_dict.items()
        except AttributeError as e:
            raise ValueError(f"Use `@register_impls` on class {cls.__name__}") from e


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
        func: NonTerminatorOpImpl[_FT, OperationInvT],
    ) -> OpImpl[_FT, OperationInvT]:
        def impl(
            ft: _FT, interpreter: Interpreter, op: OperationInvT, values: PythonValues
        ) -> OpImplResult:
            return OpImplResult(func(ft, interpreter, op, values), None)

        setattr(impl, _IMPL_OP_TYPE_KEY, op_type)
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

        setattr(impl, _IMPL_OP_TYPE_KEY, op_type)
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
        func: CastImpl[_FT, _AttributeInvT0, _AttributeInvT1],
    ) -> CastImpl[_FT, _AttributeInvT0, _AttributeInvT1]:
        setattr(func, _CAST_IMPL_TYPES_KEY, (input_type, output_type))
        return func

    return annot


AttributeInvNoDefaultT = TypeVar("AttributeInvNoDefaultT", bound=Attribute)


def impl_attr(
    input_type: type[AttributeInvNoDefaultT],
) -> Callable[
    [AttrImpl[_FT, AttributeInvNoDefaultT]],
    AttrImpl[_FT, AttributeInvNoDefaultT],
]:
    """
    Marks the conversion from an attribute to a Python value. The
    `value_for_attribute` method on `Interpreter` will call into this implementation for
    matching input and output types.

    See `InterpreterFunctions` for more documentation.
    """

    def annot(func: AttrImpl[_FT, AttributeInvT]) -> AttrImpl[_FT, AttributeInvT]:
        setattr(func, _ATTR_IMPL_TYPES_KEY, input_type)
        return func

    return annot


def impl_external(
    sym_name: str,
) -> Callable[[ExtFuncImpl[_FT]], ExtFuncImpl[_FT]]:
    """
    Marks the Python implementation of an external function.
    """

    def annot(func: ExtFuncImpl[_FT]) -> ExtFuncImpl[_FT]:
        setattr(func, _EXT_FUNC_NAME_KEY, sym_name)
        return func

    return annot


def impl_callable(
    op_type: type[OperationInvT],
) -> Callable[
    [NonTerminatorOpImpl[_FT, OperationInvT]], NonTerminatorOpImpl[_FT, OperationInvT]
]:
    """
    Marks the Python implementation of a callable operation.
    """

    if not op_type.has_trait(CallableOpInterface):
        raise ValueError("Operations that are not callable must use `impl` annotation")

    def annot(
        impl: NonTerminatorOpImpl[_FT, OperationInvT],
    ) -> NonTerminatorOpImpl[_FT, OperationInvT]:
        setattr(impl, _CALLABLE_OP_TYPE_KEY, op_type)
        return impl

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
    external_func_dict: _ExtFuncImplDict = {}
    attr_impl_dict: _AttrImplDict = {}
    callable_impl_dict: _CallableImplDict = {}

    for cls in ft.mro():
        # Iterate from subclass through superclasses
        # Assign definitions, unless they've been redefined in a subclass
        for val in cls.__dict__.values():
            if _IMPL_OP_TYPE_KEY in val.__dir__():
                # This is an annotated operation implementation
                op_type = getattr(val, _IMPL_OP_TYPE_KEY)
                if op_type not in impl_dict:
                    # subclass overrides superclass definition
                    impl_dict[op_type] = val
            elif _CAST_IMPL_TYPES_KEY in val.__dir__():
                # This is an annotated cast implementation
                types = getattr(val, _CAST_IMPL_TYPES_KEY)
                if types not in cast_impl_dict:
                    # subclass overrides superclass definition
                    cast_impl_dict[types] = val
            elif _EXT_FUNC_NAME_KEY in val.__dir__():
                # This is an annotated external function
                sym_name = getattr(val, _EXT_FUNC_NAME_KEY)
                assert isinstance(sym_name, str)
                if sym_name not in external_func_dict:
                    # subclass overrides superclass definition
                    external_func_dict[sym_name] = val
            elif _ATTR_IMPL_TYPES_KEY in val.__dir__():
                # This is an attribute value implementation
                types = getattr(val, _ATTR_IMPL_TYPES_KEY)
                if types not in attr_impl_dict:
                    # subclass overrides superclass definition
                    attr_impl_dict[types] = val
            elif _CALLABLE_OP_TYPE_KEY in val.__dir__():
                op_type = getattr(val, _CALLABLE_OP_TYPE_KEY)
                if op_type not in callable_impl_dict:
                    # subclass overrides superclass definition
                    callable_impl_dict[op_type] = val

    setattr(ft, _IMPL_DICT_KEY, impl_dict)
    setattr(ft, _CAST_IMPL_DICT_KEY, cast_impl_dict)
    setattr(ft, _ATTR_IMPL_DICT_KEY, attr_impl_dict)
    setattr(ft, _EXT_FUNC_DICT_KEY, external_func_dict)
    setattr(ft, _CALLABLE_IMPL_DICT_KEY, callable_impl_dict)

    return ft


@dataclass
class _InterpreterFunctionImpls:
    """
    Used to combine multiple function implementations. The operation
    implementations need to be passed the instance of the Functions class,
    so we keep a `(Functions, OpImpl)` tuple for every Operation type.
    """

    _impl_dict: _IMPL_DICT = field(default_factory=_IMPL_DICT)
    _cast_impl_dict: _CAST_IMPL_DICT = field(default_factory=_CAST_IMPL_DICT)
    _attr_impl_dict: _ATTR_IMPL_DICT = field(default_factory=_ATTR_IMPL_DICT)
    _external_funcs_dict: _EXT_FUNC_IMPL_DICT = field(
        default_factory=_EXT_FUNC_IMPL_DICT
    )
    _callable_impl_dict: _CALLABLE_IMPL_DICT = field(
        default_factory=_CALLABLE_IMPL_DICT
    )

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
        for types, cast_impl in cast_impls:
            if types in self._cast_impl_dict and not override:
                raise ValueError(
                    "Attempting to register implementation for cast with types "
                    f"{types}, but types already registered"
                )

            self._cast_impl_dict[types] = (ft, cast_impl)

        cast_impls = ft._attr_impls()  # pyright: ignore[reportPrivateUsage]
        for types, cast_impl in cast_impls:
            if types in self._attr_impl_dict and not override:
                raise ValueError(
                    "Attempting to register implementation for cast with types "
                    f"{types}, but types already registered"
                )

            self._attr_impl_dict[types] = (ft, cast_impl)

        ext_impls = ft._ext_impls()  # pyright: ignore[reportPrivateUsage]
        for sym_name, ext_impl in ext_impls:
            if sym_name in self._external_funcs_dict and not override:
                raise ValueError(
                    "Attempting to register external function with name "
                    f"{sym_name}, but it's already registered"
                )

            self._external_funcs_dict[sym_name] = (ft, ext_impl)

        callable_impls = ft._callable_impls()  # pyright: ignore[reportPrivateUsage]
        for op_type, impl in callable_impls:
            if op_type in self._callable_impl_dict and not override:
                raise ValueError(
                    "Attempting to register implementation for op of type "
                    f"{op_type}, but type already registered"
                )

            self._callable_impl_dict[op_type] = (ft, impl)

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

    def attr_value(
        self, interpreter: Interpreter, attr: Attribute, type_attr: Attribute
    ) -> Any:
        attr_type = type(type_attr)
        if attr_type not in self._attr_impl_dict:
            raise InterpretationError(
                f"Could not find Python value implementation for types {attr_type}"
            )
        ft, impl = self._attr_impl_dict[attr_type]
        return impl(ft, interpreter, attr, type_attr)

    def call_external(
        self, interpreter: Interpreter, sym_name: str, op: Operation, args: PythonValues
    ) -> PythonValues:
        if sym_name not in self._external_funcs_dict:
            raise InterpretationError(
                f"Could not find external function implementation named {sym_name}"
            )

        ft, ext_func = self._external_funcs_dict[sym_name]

        return ext_func(ft, interpreter, op, args)

    def call(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        ft, ext_func = self._callable_impl_dict[type(op)]

        return ext_func(ft, interpreter, op, args)


def _get_system_bitwidth() -> Literal[32, 64] | None:
    match platform.architecture()[0]:
        case "64bit":
            return 64
        case "32bit":
            return 32
        case _:
            return None


@dataclass
class Interpreter:
    """
    An extensible interpreter, initialised with a Module to interpret. The
    implementation for each Operation subclass should be provided via a
    `InterpretationFunctions` instance. Interpretations can be overridden, and
    the override must be specified explicitly, by passing `override=True` to
    the `register_functions` method.
    """

    class Listener:
        """
        Base class for observing the operations that are interpreted during a run.
        """

        def will_interpret_op(self, op: Operation, args: PythonValues) -> None: ...

        def did_interpret_op(self, op: Operation, results: PythonValues) -> None: ...

    SYSTEM_BITWIDTH: ClassVar[Literal[32, 64] | None] = _get_system_bitwidth()
    DEFAULT_BITWIDTH: ClassVar[Literal[32, 64]] = (
        32 if SYSTEM_BITWIDTH is None else SYSTEM_BITWIDTH
    )

    module: ModuleOp
    index_bitwidth: Literal[32, 64] = field(default=DEFAULT_BITWIDTH)
    """
    Number of bits in the binary representation of the index
    """
    _impls: _InterpreterFunctionImpls = field(default_factory=_InterpreterFunctionImpls)
    _ctx: ScopedDict[SSAValue, Any] = field(
        default_factory=lambda: ScopedDict[SSAValue, Any](name="root")
    )
    """
    Object holding the Python values associated with SSAValues during an
    interpretation context. An environment is a stack of scopes, values are
    assigned to the current scope, but can be fetched from a parent scope.
    """
    file: IO[str] | None = field(default=None)
    _symbol_table: dict[str, Operation] | None = None
    _impl_data: _IMPL_DATA = field(default_factory=_IMPL_DATA)
    """
    Runtime data associated with an interpreter functions implementation.
    """
    listeners: tuple[Listener, ...] = field(default=())

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

    def push_scope(self, name: str | None = None) -> None:
        """
        Create new scope in current environment, with optional custom `name`.
        """
        self._ctx = ScopedDict(name=name, parent=self._ctx)

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

    def _run_op(self, op: Operation, inputs: PythonValues) -> OpImplResult:
        if (operands_count := len(op.operands)) != (inputs_count := len(inputs)):
            raise InterpretationError(
                f"Number of operands ({operands_count}) doesn't match the number of inputs ({inputs_count})."
            )
        for listener in self.listeners:
            listener.will_interpret_op(op, inputs)
        result = self._impls.run(self, op, inputs)
        if (results_count := len(op.results)) != (
            actual_result_count := len(result.values)
        ):
            raise InterpretationError(
                f"Number of operation results ({results_count}) doesn't match the "
                f"number of implementation results ({actual_result_count})."
            )
        for listener in self.listeners:
            listener.did_interpret_op(op, result.values)
        return result

    def run_op(
        self, op: Operation | str | SymbolRefAttr, inputs: PythonValues = ()
    ) -> PythonValues:
        """
        Calls the implementation for the given operation.
        """
        if not isinstance(op, Operation):
            op = self.get_op_for_symbol(op)

        return self._run_op(op, inputs).values

    def call_op(
        self, op: Operation | str | SymbolRefAttr, inputs: PythonValues = ()
    ) -> PythonValues:
        """
        Calls the implementation for the given operation.
        """
        if not isinstance(op, Operation):
            op = self.get_op_for_symbol(op)
        results = self._impls.call(self, op, inputs)
        return results

    def call_external(
        self, sym_name: str, op: Operation, inputs: PythonValues = ()
    ) -> PythonValues:
        return self._impls.call_external(self, sym_name, op, inputs)

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

        initial_scope = self._ctx
        block = region.blocks.first

        while block is not None:
            self.push_scope(name)
            self.set_values(zip(block.args, args))

            op: Operation | None = block.first_op
            block = None

            while op is not None:
                inputs = self.get_values(op.operands)
                result = self._run_op(op, inputs)
                self.interpreter_assert(
                    len(op.results) == len(result.values),
                    f"Incorrect number of results for op {op.name}, expected {len(op.results)} but got {len(result.values)}",
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

        self._ctx = initial_scope
        return results

    def cast_value(self, o: Attribute, r: Attribute, value: Any) -> Any:
        """
        If the type of the operand and result are not the same, then look up the
        user-provided conversion function.
        """
        if o == r:
            return value

        return self._impls.cast(o, r, value)

    def value_for_attribute(self, attr: Attribute, type_attr: Attribute) -> Any:
        return self._impls.attr_value(self, attr, type_attr)

    def get_op_for_symbol(self, symbol: str | SymbolRefAttr) -> Operation:
        op = SymbolTable.lookup_symbol(self.module, symbol)
        if op is not None:
            return op
        raise InterpretationError(f"Could not find symbol {symbol}")

    def get_data(
        self,
        functions: type[InterpreterFunctions],
        key: str,
        factory: Callable[[], Any],
    ) -> Any:
        """
        Get data associated with a specific interpreter functions class, with a given key.
        If the data is missing, the `factory` argument will be executed and stored on the
        interpreter.
        """
        if functions not in self._impl_data:
            functions_data: dict[str, Any] = {}
            self._impl_data[functions] = functions_data
        else:
            functions_data = self._impl_data[functions]

        if key not in functions_data:
            data = factory()
            functions_data[key] = data
        else:
            data = functions_data[key]

        return data

    def set_data(
        self,
        functions: type[InterpreterFunctions],
        key: str,
        value: Any,
    ):
        if functions not in self._impl_data:
            functions_data: dict[str, Any] = {}
            self._impl_data[functions] = functions_data
        else:
            functions_data = self._impl_data[functions]
        functions_data[key] = value

    def print(self, *args: Any, **kwargs: Any):
        """Print to current file."""
        print(*args, **kwargs, file=self.file)

    def interpreter_assert(self, condition: bool, message: str | None = None):
        """Raise InterpretationError if condition is not satisfied."""
        if not condition:
            self.raise_error(message)

    def scope_names(self):
        ctx = self._ctx

        while ctx is not None:
            yield ctx.name or "unknown"
            ctx = ctx.parent

    def raise_error(self, message: str | None = None):
        scope_description = "/".join(self.scope_names())
        raise InterpretationError(f"AssertionError: ({scope_description})({message})")


@dataclass
class OpCounter(Interpreter.Listener):
    """
    Counts the number of times that an op has been run by the interpreter.
    """

    ops: Counter[str] = field(default_factory=Counter[str])

    def will_interpret_op(self, op: Operation, args: PythonValues) -> None:
        self.ops[op.name] += 1
