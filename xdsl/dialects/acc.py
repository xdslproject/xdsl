"""
The OpenACC (acc) dialect that models the OpenACC programming model in MLIR.

OpenACC is a directive-based programming model for accelerating applications
on heterogeneous systems. This dialect exposes compute constructs, data
constructs, loops, and the associated clauses so that host and accelerator
code can be represented, analysed, and lowered to target-specific runtimes.

See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/).
"""

from abc import ABC
from collections.abc import Hashable, Iterable, Sequence
from typing import cast

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    I1,
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
    i1,
    i32,
)
from xdsl.dialects.utils import AbstractYieldOperation, BitEnumAttribute
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    Operation,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.irdl.declarative_assembly_format import (
    AttributeVariable,
    CustomDirective,
    OperandVariable,
    OptionalOperandVariable,
    ParsingState,
    PrintingState,
    RegionVariable,
    TypeDirective,
    VariadicOperandVariable,
    irdl_custom_directive,
)
from xdsl.parser import AttrParser, Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    NoMemoryEffect,
    NoTerminator,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
    SymbolOpInterface,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


class DeviceType(StrEnum):
    NONE = "none"
    STAR = "star"
    DEFAULT = "default"
    HOST = "host"
    MULTICORE = "multicore"
    NVIDIA = "nvidia"
    RADEON = "radeon"


class ClauseDefaultValue(StrEnum):
    PRESENT = "present"
    NONE = "none"


class DataClause(StrEnum):
    """OpenACC data clause names (decomposed form).

    The original OpenACC `copy`/`copyout`/`copyin(readonly)`/etc. clauses
    are decomposed into individual `acc` ops. This enum keeps track of
    which user-level clause an op was generated for. See upstream
    `mlir::acc::DataClause`.
    """

    ACC_COPYIN = "acc_copyin"
    ACC_COPYIN_READONLY = "acc_copyin_readonly"
    ACC_COPY = "acc_copy"
    ACC_COPYOUT = "acc_copyout"
    ACC_COPYOUT_ZERO = "acc_copyout_zero"
    ACC_PRESENT = "acc_present"
    ACC_CREATE = "acc_create"
    ACC_CREATE_ZERO = "acc_create_zero"
    ACC_DELETE = "acc_delete"
    ACC_ATTACH = "acc_attach"
    ACC_DETACH = "acc_detach"
    ACC_NO_CREATE = "acc_no_create"
    ACC_PRIVATE = "acc_private"
    ACC_FIRSTPRIVATE = "acc_firstprivate"
    ACC_DEVICEPTR = "acc_deviceptr"
    ACC_GETDEVICEPTR = "acc_getdeviceptr"
    ACC_UPDATE_HOST = "acc_update_host"
    ACC_UPDATE_SELF = "acc_update_self"
    ACC_UPDATE_DEVICE = "acc_update_device"
    ACC_USE_DEVICE = "acc_use_device"
    ACC_REDUCTION = "acc_reduction"
    ACC_DECLARE_DEVICE_RESIDENT = "acc_declare_device_resident"
    ACC_DECLARE_LINK = "acc_declare_link"
    ACC_CACHE = "acc_cache"
    ACC_CACHE_READONLY = "acc_cache_readonly"


class DataClauseModifier(StrEnum):
    """Bit flags carried by a data clause op (zero / readonly / always*).

    See upstream `mlir::acc::DataClauseModifier`. Modeled as a bit-enum;
    the empty set is rendered as `none`.
    """

    ZERO = "zero"
    READONLY = "readonly"
    ALWAYSIN = "alwaysin"
    ALWAYSOUT = "alwaysout"
    CAPTURE = "capture"


class VariableTypeCategory(StrEnum):
    """Bit flags describing the OpenACC type category of a variable.

    See upstream `mlir::acc::VariableTypeCategory`. Used by the
    `MappableTypeInterface`/`PointerLikeTypeInterface` machinery; the
    empty set is rendered as `uncategorized`.
    """

    SCALAR = "scalar"
    ARRAY = "array"
    COMPOSITE = "composite"
    NONSCALAR = "nonscalar"


class ReductionOpKind(StrEnum):
    """Built-in reduction operators supported by OpenACC.

    See upstream `mlir::acc::ReductionOperator` (renamed in xDSL to avoid
    clashing with the `Operator` / `Operation` naming used elsewhere).
    Carried on `acc.reduction.recipe` to identify which reduction the recipe
    encodes.
    """

    NONE = "none"
    ADD = "add"
    MUL = "mul"
    MAX = "max"
    MIN = "min"
    IAND = "iand"
    IOR = "ior"
    XOR = "xor"
    EQV = "eqv"
    NEQV = "neqv"
    LAND = "land"
    LOR = "lor"


class LoopParMode(StrEnum):
    """Loop parallelism determination mode for `acc.loop` builders.

    See upstream `mlir::acc::LoopParMode`. Used by Python builders to pick
    between the `seq` / `independent` / `auto` attributes; the enum itself
    is not stored on the op.
    """

    SEQ = "loop_seq"
    AUTO = "loop_auto"
    INDEPENDENT = "loop_independent"


class GangArgType(StrEnum):
    """Differentiates `num=` / `dim=` / `static=` values inside an
    `acc.loop` `gang(...)` clause.

    See upstream `mlir::acc::GangArgType`. The string spellings match
    upstream's `Num` / `Dim` / `Static` symbol names so the printed
    `#acc.gang_arg_type<...>` attribute round-trips.
    """

    NUM = "Num"
    DIM = "Dim"
    STATIC = "Static"


class CombinedConstructsType(StrEnum):
    """Identifies which combined construct an `acc.loop` was decomposed
    from (`kernels loop`, `parallel loop`, `serial loop`).

    See upstream `mlir::acc::CombinedConstructsType`. Compute constructs
    (`acc.parallel` / `acc.serial` / `acc.kernels`) carry just a
    `combined` UnitAttr indicating they were a combined `... loop`
    construct; `acc.loop` carries the typed `CombinedConstructsTypeAttr`
    so the kind is recoverable.
    """

    KERNELS_LOOP = "kernels_loop"
    PARALLEL_LOOP = "parallel_loop"
    SERIAL_LOOP = "serial_loop"


@irdl_attr_definition
class DeviceTypeAttr(EnumAttribute[DeviceType]):
    """
    Device type attribute used to associate values of clauses with a specific
    device_type. Prints using the pretty form `#acc.device_type<value>` to
    match upstream MLIR (which defines `assemblyFormat = "`<` $value `>`"`).
    """

    name = "acc.device_type"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> DeviceType:
        with parser.in_angle_brackets():
            return parser.parse_str_enum(DeviceType)


@irdl_attr_definition
class ClauseDefaultValueAttr(
    EnumAttribute[ClauseDefaultValue], SpacedOpaqueSyntaxAttribute
):
    """
    Default clause value attribute, selecting either `none` or `present`.
    See upstream `acc.defaultvalue`.
    """

    name = "acc.defaultvalue"


@irdl_attr_definition
class DataClauseAttr(EnumAttribute[DataClause], SpacedOpaqueSyntaxAttribute):
    """
    OpenACC `#acc<data_clause ...>` attribute. Carried on every data-clause op
    so consumers can recover which user clause (e.g. `acc_copy`) the op was
    decomposed from. See upstream `acc.data_clause`.
    """

    name = "acc.data_clause"


@irdl_attr_definition
class DataClauseModifierAttr(
    BitEnumAttribute[DataClauseModifier], SpacedOpaqueSyntaxAttribute
):
    """
    Bit-enum attribute for data-clause modifiers (`zero` / `readonly` /
    `alwaysin` / `alwaysout` / `capture`). The empty set prints as `none`.
    See upstream `acc.data_clause_modifier`.
    """

    name = "acc.data_clause_modifier"
    none_value = "none"
    separator_value = ","
    delimiter_value = AttrParser.Delimiter.NONE


@irdl_attr_definition
class VariableTypeCategoryAttr(
    BitEnumAttribute[VariableTypeCategory], SpacedOpaqueSyntaxAttribute
):
    """
    Bit-enum attribute classifying a variable's type per the OpenACC spec.
    Empty set prints as `uncategorized`. See upstream
    `acc.variable_type_category`.
    """

    name = "acc.variable_type_category"
    none_value = "uncategorized"
    separator_value = ","
    delimiter_value = AttrParser.Delimiter.NONE


@irdl_attr_definition
class ReductionOpKindAttr(EnumAttribute[ReductionOpKind]):
    """
    Reduction operator attribute carried by `acc.reduction.recipe`. Prints
    using the pretty form `#acc.reduction_operator<value>` to match upstream
    MLIR (which defines `assemblyFormat = "`<` $value `>`"`). When referenced
    as `$reductionOperator` in an op assembly format, xDSL prints just the
    `<value>` parameter — matching upstream's inline spelling
    `reduction_operator <add>`.
    """

    name = "acc.reduction_operator"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> ReductionOpKind:
        with parser.in_angle_brackets():
            return parser.parse_str_enum(ReductionOpKind)


@irdl_attr_definition
class GangArgTypeAttr(EnumAttribute[GangArgType]):
    """
    Gang arg type attribute distinguishing `num=` / `dim=` / `static=` values
    inside `acc.loop`'s `gang(...)` clause. Prints using the pretty form
    `#acc.gang_arg_type<value>` to match upstream MLIR (which defines
    `assemblyFormat = "`<` $value `>`"`).
    """

    name = "acc.gang_arg_type"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> GangArgType:
        with parser.in_angle_brackets():
            return parser.parse_str_enum(GangArgType)


@irdl_attr_definition
class CombinedConstructsTypeAttr(EnumAttribute[CombinedConstructsType]):
    """
    Combined-constructs attribute carried on `acc.loop` to identify the
    user-level `kernels loop` / `parallel loop` / `serial loop` it was
    decomposed from. Defaulted to no value when the loop stands alone.
    Prints using the pretty form `#acc.combined_constructs<value>` to
    match upstream MLIR (whose `EnumAttr` default
    `assemblyFormat = "`<` $value `>`"` produces the dot form, *not* the
    spaced-opaque form `#acc<combined_constructs ...>`).
    """

    name = "acc.combined_constructs"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> CombinedConstructsType:
        with parser.in_angle_brackets():
            return parser.parse_str_enum(CombinedConstructsType)


@irdl_attr_definition
class DataBoundsType(ParametrizedAttribute, TypeAttribute):
    """
    Type used for `acc.bounds` results. Holds normalized bounds information
    for an `acc` data clause; consumed by the data-clause ops via their
    variadic `bounds` operand and by the `acc.get_*bound`/`get_extent`/
    `get_stride` accessor ops.
    See upstream `!acc.data_bounds_ty`.
    """

    name = "acc.data_bounds_ty"


@irdl_attr_definition
class DeclareTokenType(ParametrizedAttribute, TypeAttribute):
    """
    Type returned by `acc.declare_enter` and consumed by `acc.declare_exit`
    to represent an implicit OpenACC data region. See upstream
    `!acc.declare_token`.
    """

    name = "acc.declare_token"


def _parse_device_type_attr(parser: Parser) -> DeviceTypeAttr:
    """Parse a single `#acc.device_type<...>` attribute."""
    attr = parser.parse_attribute()
    if not isinstance(attr, DeviceTypeAttr):
        parser.raise_error("expected #acc.device_type attribute")
    return attr


def _parse_optional_device_type_suffix(parser: Parser) -> DeviceTypeAttr:
    """Parse an optional `[ #acc.device_type<...> ]`, defaulting to `#none`."""
    if parser.parse_optional_punctuation("[") is None:
        return DeviceTypeAttr(DeviceType.NONE)
    dt = _parse_device_type_attr(parser)
    parser.parse_punctuation("]")
    return dt


def _parse_typed_operand(parser: Parser) -> tuple[UnresolvedOperand, Attribute]:
    """Parse `%v : <type>` and return `(operand, type)`."""
    operand = parser.parse_unresolved_operand()
    parser.parse_punctuation(":")
    return operand, parser.parse_type()


def _parse_operand_with_dt(
    parser: Parser,
) -> tuple[UnresolvedOperand, Attribute, DeviceTypeAttr]:
    """Parse `%v : <type> ( `[` #acc.device_type<...> `]` )?`."""
    operand, ty = _parse_typed_operand(parser)
    return operand, ty, _parse_optional_device_type_suffix(parser)


def _print_device_type_suffix(printer: Printer, dt: Attribute) -> None:
    """Print ` [#acc.device_type<...>]` unless the device type is `#none`."""
    if isinstance(dt, DeviceTypeAttr) and dt.data == DeviceType.NONE:
        return
    printer.print_string(" [")
    printer.print_attribute(dt)
    printer.print_string("]")


def _print_typed_operand(printer: Printer, operand: SSAValue) -> None:
    """Print `%v : <type>`."""
    printer.print_ssa_value(operand)
    printer.print_string(" : ")
    printer.print_attribute(operand.type)


def _emit_clause_keyword(printer: Printer, state: PrintingState, keyword: str) -> None:
    """Print whitespace + a clause keyword and update print state so the
    body's first character (typically ``(``) lands adjacent (e.g.
    ``async(``). Used by the three clauses of `KernelEnvironmentClauses`."""
    state.print_whitespace(printer)
    printer.print_string(keyword)
    state.should_emit_space = True
    state.last_was_punctuation = False


def _print_operand_with_dt(printer: Printer, operand: SSAValue, dt: Attribute) -> None:
    """Print `%v : <type> ( [#acc.device_type<...>] )?`."""
    _print_typed_operand(printer, operand)
    _print_device_type_suffix(printer, dt)


_DEVICE_TYPE_ONLY_NONE = ArrayAttr((DeviceTypeAttr(DeviceType.NONE),))
"""Upstream `hasOnlyDeviceTypeNone` sentinel: the `[#acc.device_type<none>]` array."""


def _parse_num_gangs_group(
    parser: Parser,
) -> tuple[tuple[UnresolvedOperand, ...], tuple[Attribute, ...], DeviceTypeAttr]:
    """Parse one `{ %v : type, ... } ( `[` #acc.device_type<...> `]` )?` group."""
    with parser.in_braces():
        pairs = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, lambda: _parse_typed_operand(parser)
        )
    operands, types = zip(*pairs)
    return operands, types, _parse_optional_device_type_suffix(parser)


def _parse_wait_group(
    parser: Parser,
) -> tuple[
    tuple[UnresolvedOperand, ...],
    tuple[Attribute, ...],
    DeviceTypeAttr,
    BoolAttr,
]:
    """Parse one `{ (`devnum:`)? %v : type, ... } ( `[` #acc.device_type<...> `]` )?` group."""
    with parser.in_braces():
        has_devnum = parser.parse_optional_keyword("devnum") is not None
        if has_devnum:
            parser.parse_punctuation(":")
        pairs = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, lambda: _parse_typed_operand(parser)
        )
    operands, types = zip(*pairs)
    return (
        operands,
        types,
        _parse_optional_device_type_suffix(parser),
        IntegerAttr.from_bool(has_devnum),
    )


def _flatten_groups(
    groups: Sequence[
        tuple[Sequence[UnresolvedOperand], Sequence[Attribute], DeviceTypeAttr]
    ],
) -> tuple[
    tuple[UnresolvedOperand, ...],
    tuple[Attribute, ...],
    tuple[DeviceTypeAttr, ...],
    tuple[int, ...],
]:
    """Flatten a list of parsed groups into operands / types / dts / segment sizes."""
    operands: list[UnresolvedOperand] = []
    types: list[Attribute] = []
    dts: list[DeviceTypeAttr] = []
    segs: list[int] = []
    for group_operands, group_types, dt in groups:
        operands.extend(group_operands)
        types.extend(group_types)
        dts.append(dt)
        segs.append(len(group_operands))
    return tuple(operands), tuple(types), tuple(dts), tuple(segs)


def _print_groups(
    printer: Printer,
    operands: Sequence[SSAValue],
    device_types: Sequence[Attribute],
    segments: Sequence[int],
    *,
    devnum_flags: Sequence[Attribute] = (),
) -> None:
    """Print brace-grouped operands `{ ( `devnum:` )? %v : T, ... }[dt]`, comma-separated."""
    idx = 0
    for group_idx, (size, dt) in enumerate(zip(segments, device_types, strict=True)):
        if group_idx:
            printer.print_string(", ")
        printer.print_string("{")
        if group_idx < len(devnum_flags):
            flag = devnum_flags[group_idx]
            if isinstance(flag, IntegerAttr) and bool(flag.value.data):
                printer.print_string("devnum: ")
        for i in range(size):
            if i:
                printer.print_string(", ")
            _print_typed_operand(printer, operands[idx])
            idx += 1
        printer.print_string("}")
        _print_device_type_suffix(printer, dt)


@irdl_custom_directive
class DeviceTypeOperands(CustomDirective):
    """Port of upstream `custom<DeviceTypeOperands>`.

    Syntax inside the enclosing `(`...`)`:
      `%op : type ( `[` #acc.device_type<...> `]` )?` (`,` ...)*
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    device_types: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_optional_like(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(self.operands.get(op))

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        triples = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, lambda: _parse_operand_with_dt(parser)
        )
        operands, types, device_types = zip(*triples)
        self.operands.set(state, operands)
        self.operand_types.set(state, types)
        self.device_types.set(state, ArrayAttr(device_types))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        if not operands:
            return
        dts = (
            attr.data
            if isa(attr := self.device_types.get(op), ArrayAttr)
            else (DeviceTypeAttr(DeviceType.NONE),) * len(operands)
        )
        printer.print_list(
            zip(operands, dts, strict=True),
            lambda pair: _print_operand_with_dt(printer, pair[0], pair[1]),
        )
        state.should_emit_space = True
        state.last_was_punctuation = False


def _parse_dt_kw_only_body(
    parser: Parser,
) -> tuple[
    Sequence[UnresolvedOperand],
    Sequence[Attribute],
    ArrayAttr[DeviceTypeAttr] | None,
    ArrayAttr[DeviceTypeAttr] | None,
]:
    """Parse the post-keyword body of an `async`-style clause.

    Returns ``(operands, types, device_types, keyword_only)``. Either or
    both attributes may be ``None`` when no list / no operands are present.
    The bare-keyword form (no parens) yields ``keyword_only =
    [#acc.device_type<none>]``. Shared between
    ``DeviceTypeOperandsWithKeywordOnly`` and ``DataEntryOilist``'s
    ``async`` branch.
    """
    if not parser.parse_optional_punctuation("("):
        return (), (), None, ArrayAttr([DeviceTypeAttr(DeviceType.NONE)])

    kw_only = parser.parse_optional_comma_separated_list(
        parser.Delimiter.SQUARE, lambda: _parse_device_type_attr(parser)
    )
    keyword_only = ArrayAttr(kw_only) if kw_only is not None else None

    if parser.parse_optional_punctuation(")"):
        return (), (), None, keyword_only

    if keyword_only is not None:
        parser.parse_punctuation(",")

    triples = parser.parse_comma_separated_list(
        parser.Delimiter.NONE, lambda: _parse_operand_with_dt(parser)
    )
    parser.parse_punctuation(")")
    operands, types, device_types = zip(*triples)
    return operands, types, ArrayAttr(device_types), keyword_only


def _print_dt_kw_only_body(
    printer: Printer,
    state: PrintingState,
    operands: Sequence[SSAValue],
    keyword_only: Attribute | None,
    device_types: Attribute | None,
) -> None:
    """Print the post-keyword body of an `async`-style clause.

    Mirror of `_parse_dt_kw_only_body`. Emits nothing when both
    ``operands`` is empty and ``keyword_only`` is the all-`#none` sentinel
    — matching upstream's "bare async keyword" elision.
    """
    if not operands and keyword_only == _DEVICE_TYPE_ONLY_NONE:
        return

    printer.print_string("(")
    if isa(keyword_only, ArrayAttr) and keyword_only.data:
        printer.print_string("[")
        printer.print_list(keyword_only.data, printer.print_attribute)
        printer.print_string("]")
        if operands:
            printer.print_string(", ")
    if operands:
        dts = (
            attr.data
            if isa(attr := device_types, ArrayAttr)
            else (DeviceTypeAttr(DeviceType.NONE),) * len(operands)
        )
        printer.print_list(
            zip(operands, dts, strict=True),
            lambda pair: _print_operand_with_dt(printer, pair[0], pair[1]),
        )
    printer.print_string(")")
    state.should_emit_space = True
    state.last_was_punctuation = False


@irdl_custom_directive
class DeviceTypeOperandsWithKeywordOnly(CustomDirective):
    """Port of upstream `custom<DeviceTypeOperandsWithKeywordOnly>`.

    Follows a bare keyword (e.g. `async`) in the format. The directive owns
    the optional surrounding parentheses. Syntax options after the keyword:
      bare                         → keyword-only = [#none]
      `(` `[` dts `]` `)`          → keyword-only list, no operands
      `(` operand-list `)`         → operands with optional per-operand dt
      `(` `[` dts `]` `,` ops `)`  → mix of keyword-only and operands
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    device_types: AttributeVariable
    keyword_only: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_optional_like(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return (
            bool(self.operands.get(op))
            or self.keyword_only.get(op) is not None
            or self.device_types.get(op) is not None
        )

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        operands, types, device_types, keyword_only = _parse_dt_kw_only_body(parser)
        self.operands.set(state, operands)
        self.operand_types.set(state, types)
        if device_types is not None:
            self.device_types.set(state, device_types)
        if keyword_only is not None:
            self.keyword_only.set(state, keyword_only)
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        _print_dt_kw_only_body(
            printer,
            state,
            self.operands.get(op),
            self.keyword_only.get(op),
            self.device_types.get(op),
        )


@irdl_custom_directive
class NumGangs(CustomDirective):
    """Port of upstream `custom<NumGangs>`.

    Groups of operands with a per-group device type and a segments array.
    Syntax inside the enclosing `(`...`)`:
      `{` op:type (`,` op:type)* `}` (`[` dt `]`)?  (`,` ...)*
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    device_types: AttributeVariable
    segments: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_optional_like(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(self.operands.get(op))

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        groups = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, lambda: _parse_num_gangs_group(parser)
        )
        operands, types, dts, segs = _flatten_groups(groups)
        self.operands.set(state, operands)
        self.operand_types.set(state, types)
        self.device_types.set(state, ArrayAttr(dts))
        self.segments.set(state, DenseArrayBase.from_list(i32, segs))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        if not operands:
            return
        dts = (
            attr.data
            if isa(attr := self.device_types.get(op), ArrayAttr)
            else (DeviceTypeAttr(DeviceType.NONE),)
        )
        seg_values: Sequence[int] = (
            segments.get_values()
            if isinstance(segments := self.segments.get(op), DenseArrayBase)
            else (len(operands),)
        )
        _print_groups(printer, operands, dts, seg_values)
        state.should_emit_space = True
        state.last_was_punctuation = False


def _parse_wait_body(
    parser: Parser,
) -> tuple[
    Sequence[UnresolvedOperand],
    Sequence[Attribute],
    ArrayAttr[DeviceTypeAttr] | None,
    DenseArrayBase | None,
    ArrayAttr[BoolAttr] | None,
    ArrayAttr[DeviceTypeAttr] | None,
]:
    """Parse the post-keyword body of a `wait` clause.

    Returns ``(operands, types, device_types, segments, has_devnum,
    keyword_only)``. The bare-keyword form (no parens) yields
    ``keyword_only = [#acc.device_type<none>]`` and all other slots
    ``None`` / empty. Shared between :class:`WaitClause` and the
    :class:`KernelEnvironmentClauses` oilist.
    """
    if not parser.parse_optional_punctuation("("):
        return (
            (),
            (),
            None,
            None,
            None,
            ArrayAttr([DeviceTypeAttr(DeviceType.NONE)]),
        )

    kw_only = parser.parse_optional_comma_separated_list(
        parser.Delimiter.SQUARE, lambda: _parse_device_type_attr(parser)
    )
    keyword_only = ArrayAttr(kw_only) if kw_only is not None else None
    if keyword_only is not None:
        parser.parse_punctuation(",")

    groups = parser.parse_comma_separated_list(
        parser.Delimiter.NONE, lambda: _parse_wait_group(parser)
    )
    parser.parse_punctuation(")")

    all_operands: list[UnresolvedOperand] = []
    all_types: list[Attribute] = []
    dts: list[DeviceTypeAttr] = []
    devnum_flags: list[BoolAttr] = []
    segs: list[int] = []
    for group_operands, group_types, dt, devnum in groups:
        all_operands.extend(group_operands)
        all_types.extend(group_types)
        dts.append(dt)
        devnum_flags.append(devnum)
        segs.append(len(group_operands))

    return (
        tuple(all_operands),
        tuple(all_types),
        ArrayAttr(dts),
        DenseArrayBase.from_list(i32, segs),
        ArrayAttr(devnum_flags),
        keyword_only,
    )


def _print_wait_body(
    printer: Printer,
    state: PrintingState,
    operands: Sequence[SSAValue],
    keyword_only: Attribute | None,
    device_types: Attribute | None,
    segments: Attribute | None,
    has_devnum: Attribute | None,
) -> None:
    """Print the post-keyword body of a `wait` clause.

    Mirror of :func:`_parse_wait_body`. Emits nothing when both ``operands``
    is empty and ``keyword_only`` is the all-`#none` sentinel — matching
    upstream's "bare wait keyword" elision.
    """
    if not operands and keyword_only == _DEVICE_TYPE_ONLY_NONE:
        return

    printer.print_string("(")
    if (
        isa(keyword_only, ArrayAttr)
        and keyword_only.data
        and keyword_only != _DEVICE_TYPE_ONLY_NONE
    ):
        printer.print_string("[")
        printer.print_list(keyword_only.data, printer.print_attribute)
        printer.print_string("]")
        if operands:
            printer.print_string(", ")
    if operands:
        dts = (
            attr.data
            if isa(attr := device_types, ArrayAttr)
            else (DeviceTypeAttr(DeviceType.NONE),)
        )
        seg_values: Sequence[int] = (
            segments.get_values()
            if isinstance(segments, DenseArrayBase)
            else (len(operands),)
        )
        devnum_flags: Sequence[Attribute] = (
            has_devnum.data if isa(has_devnum, ArrayAttr) else ()
        )
        _print_groups(printer, operands, dts, seg_values, devnum_flags=devnum_flags)
    printer.print_string(")")
    state.should_emit_space = True
    state.last_was_punctuation = False


@irdl_custom_directive
class WaitClause(CustomDirective):
    """Port of upstream `custom<WaitClause>`.

    Follows a bare `wait` keyword in the format. The directive owns the
    optional surrounding parentheses. Syntax options after the keyword:
      bare                                                → keyword-only = [#none]
      `(` `[` dts `]` `,` group (`,` group)* `)`          → kw-only + operand groups
      `(` group (`,` group)* `)`                          → operand groups only
    where each group is `{ (`devnum:`)? %v : type, ... } (`[` dt `]`)?`.

    The post-keyword body parse/print is shared via :func:`_parse_wait_body`
    / :func:`_print_wait_body` so other oilist-style directives can reuse
    the same spelling.
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    device_types: AttributeVariable
    segments: AttributeVariable
    has_devnum: AttributeVariable
    keyword_only: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_optional_like(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return (
            bool(self.operands.get(op))
            or self.keyword_only.get(op) is not None
            or self.device_types.get(op) is not None
        )

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        operands, types, dts, segs, devnum, kw_only = _parse_wait_body(parser)
        self.operands.set(state, operands)
        self.operand_types.set(state, types)
        if dts is not None:
            self.device_types.set(state, dts)
        if segs is not None:
            self.segments.set(state, segs)
        if devnum is not None:
            self.has_devnum.set(state, devnum)
        if kw_only is not None:
            self.keyword_only.set(state, kw_only)
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        _print_wait_body(
            printer,
            state,
            self.operands.get(op),
            self.keyword_only.get(op),
            self.device_types.get(op),
            self.segments.get(op),
            self.has_devnum.get(op),
        )


@irdl_custom_directive
class KernelEnvironmentClauses(CustomDirective):
    """Port of upstream `acc.kernel_environment`'s `oilist(...)`.

    Accepts the three optional clauses `dataOperands` / `async` / `wait` in
    any order on parse; emits them in upstream's td-definition order on
    print (data, async, wait). Each clause may appear at most once. The
    body parsers/printers for `async` and `wait` are shared with
    :class:`DeviceTypeOperandsWithKeywordOnly` and :class:`WaitClause` via
    the `_parse/_print_dt_kw_only_body` and `_parse/_print_wait_body`
    helpers, so the spelling is bit-identical.
    """

    data_clause_operands: VariadicOperandVariable
    data_clause_operand_types: TypeDirective
    async_operands: VariadicOperandVariable
    async_operand_types: TypeDirective
    async_device_types: AttributeVariable
    async_only: AttributeVariable
    wait_operands: VariadicOperandVariable
    wait_operand_types: TypeDirective
    wait_device_types: AttributeVariable
    wait_segments: AttributeVariable
    wait_has_devnum: AttributeVariable
    wait_only: AttributeVariable

    def is_optional_like(self) -> bool:
        return True

    def set_empty(self, state: ParsingState) -> None:
        self.data_clause_operands.set(state, ())
        self.data_clause_operand_types.set(state, ())
        self.async_operands.set(state, ())
        self.async_operand_types.set(state, ())
        self.wait_operands.set(state, ())
        self.wait_operand_types.set(state, ())

    _CLAUSE_KEYWORDS = ("dataOperands", "async", "wait")

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        self.set_empty(state)
        seen: set[str] = set()
        while (
            kw := parser.parse_optional_keyword_in(self._CLAUSE_KEYWORDS)
        ) is not None:
            if kw in seen:
                parser.raise_error(f"'{kw}' clause specified twice")
            seen.add(kw)
            if kw == "dataOperands":
                with parser.in_parens():
                    pairs = parser.parse_comma_separated_list(
                        parser.Delimiter.NONE, lambda: _parse_typed_operand(parser)
                    )
                operands, types = zip(*pairs)
                self.data_clause_operands.set(state, operands)
                self.data_clause_operand_types.set(state, types)
            elif kw == "async":
                ops, types, dts, kw_only = _parse_dt_kw_only_body(parser)
                self.async_operands.set(state, ops)
                self.async_operand_types.set(state, types)
                if dts is not None:
                    self.async_device_types.set(state, dts)
                if kw_only is not None:
                    self.async_only.set(state, kw_only)
            else:  # wait
                ops, types, dts, segs, devnum, kw_only = _parse_wait_body(parser)
                self.wait_operands.set(state, ops)
                self.wait_operand_types.set(state, types)
                if dts is not None:
                    self.wait_device_types.set(state, dts)
                if segs is not None:
                    self.wait_segments.set(state, segs)
                if devnum is not None:
                    self.wait_has_devnum.set(state, devnum)
                if kw_only is not None:
                    self.wait_only.set(state, kw_only)
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if data_operands := self.data_clause_operands.get(op):
            _emit_clause_keyword(printer, state, "dataOperands")
            printer.print_string("(")
            printer.print_list(
                data_operands, lambda v: _print_typed_operand(printer, v)
            )
            printer.print_string(")")

        async_operands = self.async_operands.get(op)
        async_only = self.async_only.get(op)
        async_device_types = self.async_device_types.get(op)
        if async_operands or async_only is not None or async_device_types is not None:
            _emit_clause_keyword(printer, state, "async")
            _print_dt_kw_only_body(
                printer, state, async_operands, async_only, async_device_types
            )

        wait_operands = self.wait_operands.get(op)
        wait_only = self.wait_only.get(op)
        wait_device_types = self.wait_device_types.get(op)
        if wait_operands or wait_only is not None or wait_device_types is not None:
            _emit_clause_keyword(printer, state, "wait")
            _print_wait_body(
                printer,
                state,
                wait_operands,
                wait_only,
                wait_device_types,
                self.wait_segments.get(op),
                self.wait_has_devnum.get(op),
            )


@irdl_custom_directive
class OperandWithKeywordOnly(CustomDirective):
    """Port of upstream `custom<OperandWithKeywordOnly>`.

    Follows a bare keyword (e.g. `async`) in the format. After the keyword:
      bare                       → keyword-only UnitAttr set
      `(` operand `:` type `)`   → operand set
    """

    operand: OptionalOperandVariable
    operand_type: TypeDirective
    attr: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return self.operand.get(op) is not None or self.attr.get(op) is not None

    def set_empty(self, state: ParsingState) -> None:
        self.operand.set(state, None)
        self.operand_type.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if not parser.parse_optional_punctuation("("):
            self.attr.set(state, UnitAttr())
            self.operand.set(state, None)
            self.operand_type.set(state, ())
            return True
        operand, ty = _parse_typed_operand(parser)
        parser.parse_punctuation(")")
        self.operand.set(state, operand)
        self.operand_type.set(state, (ty,))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        # Mirror upstream's `if (attr) return;`: when the bare-keyword
        # UnitAttr is set, the parent already emitted the keyword and we
        # contribute nothing further.
        if self.attr.get(op) is not None:
            return
        operand = self.operand.get(op)
        assert operand is not None  # is_present excludes the all-None case
        printer.print_string("(")
        _print_typed_operand(printer, operand)
        printer.print_string(")")
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class OperandsWithKeywordOnly(CustomDirective):
    """Port of upstream `custom<OperandsWithKeywordOnly>`.

    Follows a bare keyword (e.g. `wait`) in the format. After the keyword:
      bare                                         → keyword-only UnitAttr
      `(` op (`,` op)* `:` ty (`,` ty)* `)`        → variadic operand list
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    attr: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(self.operands.get(op)) or self.attr.get(op) is not None

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if not parser.parse_optional_punctuation("("):
            self.attr.set(state, UnitAttr())
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True
        operands = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )
        parser.parse_punctuation(":")
        types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_type
        )
        parser.parse_punctuation(")")
        self.operands.set(state, operands)
        self.operand_types.set(state, types)
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if self.attr.get(op) is not None:
            return
        operands = self.operands.get(op)
        assert operands  # is_present excludes the empty/no-attr case
        printer.print_string("(")
        printer.print_list(operands, printer.print_ssa_value)
        printer.print_string(" : ")
        printer.print_list(
            (operand.type for operand in operands), printer.print_attribute
        )
        printer.print_string(")")
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class AtomicIfClause(CustomDirective):
    """Port of the `acc.atomic.*` family's `oilist( \\`if\\` \\`(\\` $ifCond \\`)\\` )`.

    Single-clause oilist shared by `acc.atomic.read` / `acc.atomic.write` /
    `acc.atomic.update` / `acc.atomic.capture`. When absent on parse, no
    operand is set; when present, parses `if(%cond)` and types the operand
    as `i1` implicitly (upstream `Optional<I1>:$ifCond`).
    """

    if_cond: OptionalOperandVariable
    if_cond_type: TypeDirective

    def is_optional_like(self) -> bool:
        return True

    def set_empty(self, state: ParsingState) -> None:
        self.if_cond.set(state, None)
        self.if_cond_type.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        self.set_empty(state)
        if parser.parse_optional_keyword("if") is None:
            return True
        parser.parse_punctuation("(")
        operand = parser.parse_unresolved_operand()
        parser.parse_punctuation(")")
        self.if_cond.set(state, operand)
        self.if_cond_type.set(state, (i1,))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if_cond = self.if_cond.get(op)
        if if_cond is None:
            return
        state.print_whitespace(printer)
        printer.print_string("if(")
        printer.print_ssa_value(if_cond)
        printer.print_string(")")
        state.should_emit_space = True
        state.last_was_punctuation = False


def _default_var_type(var_type: Attribute) -> Attribute:
    """Default `varType` value for a `var` operand.

    Mirrors upstream's `parseVarPtrType`/`printVarPtrType` heuristic: when
    `var` has a pointer-like type, the implied `varType` is the pointee
    element type; otherwise the variable's own type is used.
    """
    if isa(var_type, MemRefType):
        return var_type.element_type
    return var_type


@irdl_custom_directive
class AccVar(CustomDirective):
    """Port of upstream `custom<AccVar>($accVar, type($accVar))`.

    Renders `accPtr(%v : type)`. Accepts both `accPtr` and `accVar`
    keywords on parse; always emits `accPtr` (xDSL doesn't distinguish
    `PointerLikeType` vs `MappableType` here, mirroring `Var`'s `varPtr`
    choice).
    """

    acc_var: OperandVariable
    acc_var_type: TypeDirective

    def is_anchorable(self) -> bool:
        return False

    def is_optional_like(self) -> bool:
        return False

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if parser.parse_optional_keyword("accPtr") is None:
            parser.parse_keyword("accVar")
        with parser.in_parens():
            operand = parser.parse_unresolved_operand()
            parser.parse_punctuation(":")
            ty = parser.parse_type()
        self.acc_var.set(state, operand)
        self.acc_var_type.set(state, (ty,))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)
        ssa = self.acc_var.get(op)
        printer.print_string("accPtr(")
        printer.print_ssa_value(ssa)
        printer.print_string(" : ")
        printer.print_attribute(ssa.type)
        printer.print_string(")")
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class Var(CustomDirective):
    """Port of upstream `custom<Var>($var) `:` custom<VarPtrType>(type($var), $varType)`.

    Renders `varPtr(%v : type) (varType(t))?`. Accepts both `varPtr` and `var`
    keywords on parse; always emits `varPtr` (xDSL doesn't distinguish
    `PointerLikeType` vs `MappableType` here, and the OpenACC tests target
    memref types which are pointer-like). The `varType` slot is omitted on
    print whenever it matches `_default_var_type(var.type)`.
    """

    var: OperandVariable
    var_type: TypeDirective
    var_type_attr: AttributeVariable

    def is_anchorable(self) -> bool:
        return False

    def is_optional_like(self) -> bool:
        return False

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if parser.parse_optional_keyword("varPtr") is None:
            parser.parse_keyword("var")
        with parser.in_parens():
            operand = parser.parse_unresolved_operand()
            parser.parse_punctuation(":")
            ty = parser.parse_type()
        self.var.set(state, operand)
        self.var_type.set(state, (ty,))

        if parser.parse_optional_keyword("varType") is not None:
            with parser.in_parens():
                self.var_type_attr.set(state, parser.parse_type())
        else:
            self.var_type_attr.set(state, _default_var_type(ty))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)
        ssa = self.var.get(op)
        printer.print_string("varPtr(")
        printer.print_ssa_value(ssa)
        printer.print_string(" : ")
        printer.print_attribute(ssa.type)
        printer.print_string(")")
        attr = self.var_type_attr.get(op)
        if attr is not None and attr != _default_var_type(ssa.type):
            printer.print_string(" varType(")
            printer.print_attribute(attr)
            printer.print_string(")")
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class DataEntryOilist(CustomDirective):
    """Port of upstream's `oilist(...)` for the data-entry op family.

    Accepts the four optional clauses `varPtrPtr` / `bounds` / `async` /
    `recipe` in any order on parse; emits them in the canonical
    upstream-td-definition order on print. Each clause may appear at most
    once. This mirrors upstream MLIR's `oilist(...)` semantics — the
    round-trip with `mlir-opt` (which emits in upstream's td order)
    round-trips bit-identically through xDSL even when the input source
    interleaves clauses in a different order.
    """

    var_ptr_ptr: OptionalOperandVariable
    var_ptr_ptr_type: TypeDirective
    bounds: VariadicOperandVariable
    async_operands: VariadicOperandVariable
    async_operand_types: TypeDirective
    async_device_type: AttributeVariable
    async_only: AttributeVariable
    recipe: AttributeVariable

    def is_optional_like(self) -> bool:
        return True

    def set_empty(self, state: ParsingState) -> None:
        self.var_ptr_ptr.set(state, None)
        self.var_ptr_ptr_type.set(state, ())
        self.bounds.set(state, ())
        self.async_operands.set(state, ())
        self.async_operand_types.set(state, ())

    _CLAUSE_KEYWORDS = ("varPtrPtr", "bounds", "async", "recipe")

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        self.set_empty(state)
        seen: set[str] = set()
        while (
            kw := parser.parse_optional_keyword_in(self._CLAUSE_KEYWORDS)
        ) is not None:
            if kw in seen:
                parser.raise_error(f"'{kw}' clause specified twice")
            seen.add(kw)
            if kw == "varPtrPtr":
                with parser.in_parens():
                    operand = parser.parse_unresolved_operand()
                    parser.parse_punctuation(":")
                    ty = parser.parse_type()
                self.var_ptr_ptr.set(state, operand)
                self.var_ptr_ptr_type.set(state, (ty,))
            elif kw == "bounds":
                with parser.in_parens():
                    bounds_operands = parser.parse_comma_separated_list(
                        parser.Delimiter.NONE, parser.parse_unresolved_operand
                    )
                self.bounds.set(state, bounds_operands)
            elif kw == "async":
                ops, types, dts, kw_only = _parse_dt_kw_only_body(parser)
                self.async_operands.set(state, ops)
                self.async_operand_types.set(state, types)
                if dts is not None:
                    self.async_device_type.set(state, dts)
                if kw_only is not None:
                    self.async_only.set(state, kw_only)
            else:  # recipe
                with parser.in_parens():
                    self.recipe.set(state, parser.parse_attribute())
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        var_ptr_ptr = self.var_ptr_ptr.get(op)
        if var_ptr_ptr is not None:
            state.print_whitespace(printer)
            printer.print_string("varPtrPtr(")
            printer.print_ssa_value(var_ptr_ptr)
            printer.print_string(" : ")
            printer.print_attribute(var_ptr_ptr.type)
            printer.print_string(")")
            state.should_emit_space = True
            state.last_was_punctuation = False

        bounds = self.bounds.get(op)
        if bounds:
            state.print_whitespace(printer)
            printer.print_string("bounds(")
            printer.print_list(bounds, printer.print_ssa_value)
            printer.print_string(")")
            state.should_emit_space = True
            state.last_was_punctuation = False

        async_operands = self.async_operands.get(op)
        async_only = self.async_only.get(op)
        async_device_type = self.async_device_type.get(op)
        # Mirror upstream's optional-group anchor: the `async` keyword is
        # emitted whenever any of the four async slots is set, including the
        # all-`#none` sentinel (which renders bare `async`).
        async_present = (
            bool(async_operands)
            or async_only is not None
            or async_device_type is not None
        )
        if async_present:
            state.print_whitespace(printer)
            printer.print_string("async")
            # Leave `should_emit_space = True` so the bare-keyword case (body
            # returns early) lands a space before the next clause / `->`.
            # The body's `print_string("(")` doesn't consult this flag, so
            # `async(...)` still prints adjacent in the non-bare case.
            state.should_emit_space = True
            state.last_was_punctuation = False
            _print_dt_kw_only_body(
                printer, state, async_operands, async_only, async_device_type
            )

        recipe = self.recipe.get(op)
        if recipe is not None:
            state.print_whitespace(printer)
            printer.print_string("recipe(")
            printer.print_attribute(recipe)
            printer.print_string(")")
            state.should_emit_space = True
            state.last_was_punctuation = False


@irdl_custom_directive
class CombinedConstructsLoop(CustomDirective):
    """Port of upstream `custom<CombinedConstructsLoop>($combined)`.

    Sits inside `acc.loop`'s `combined ( ... )` group: parses one of the
    bare keywords `kernels` / `parallel` / `serial` and produces a
    `CombinedConstructsTypeAttr`. On print, emits the matching keyword.
    """

    combined: AttributeVariable

    _KEYWORDS = ("kernels", "parallel", "serial")
    _BY_KEYWORD = {
        "kernels": CombinedConstructsType.KERNELS_LOOP,
        "parallel": CombinedConstructsType.PARALLEL_LOOP,
        "serial": CombinedConstructsType.SERIAL_LOOP,
    }
    _BY_VALUE = {v: k for k, v in _BY_KEYWORD.items()}

    def is_anchorable(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return self.combined.get(op) is not None

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        for kw in self._KEYWORDS:
            if parser.parse_optional_keyword(kw) is not None:
                self.combined.set(
                    state, CombinedConstructsTypeAttr(self._BY_KEYWORD[kw])
                )
                return True
        parser.raise_error("expected compute construct name")

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        attr = self.combined.get(op)
        assert isinstance(attr, CombinedConstructsTypeAttr)
        printer.print_string(self._BY_VALUE[attr.data])


@irdl_custom_directive
class DeviceTypeOperandsWithSegment(CustomDirective):
    """Port of upstream `custom<DeviceTypeOperandsWithSegment>`.

    Used by `acc.loop` for the `tile(...)` clause. Groups operands inside
    `{...}`, with an optional per-group `[#acc.device_type<...>]` suffix
    and a `DenseI32ArrayAttr` segments array. Syntax inside the enclosing
    `( ... )`:
      `{` op:type (`,` op:type)* `}` (`[` dt `]`)?  (`,` ...)*
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    device_types: AttributeVariable
    segments: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(self.operands.get(op))

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        groups = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, lambda: _parse_num_gangs_group(parser)
        )
        operands, types, dts, segs = _flatten_groups(groups)
        self.operands.set(state, operands)
        self.operand_types.set(state, types)
        self.device_types.set(state, ArrayAttr(dts))
        self.segments.set(state, DenseArrayBase.from_list(i32, segs))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        # `is_present = bool(self.operands.get(op))` — when this `print` is
        # called the operand list is guaranteed non-empty by the framework's
        # anchor mechanism, so no empty-list early-return guard is needed.
        operands = self.operands.get(op)
        dts = (
            attr.data
            if isa(attr := self.device_types.get(op), ArrayAttr)
            else (DeviceTypeAttr(DeviceType.NONE),)
        )
        seg_values: Sequence[int] = (
            segments.get_values()
            if isinstance(segments := self.segments.get(op), DenseArrayBase)
            else (len(operands),)
        )
        _print_groups(printer, operands, dts, seg_values)
        state.should_emit_space = True
        state.last_was_punctuation = False


_GANG_KEYWORDS: tuple[tuple[str, GangArgType], ...] = (
    ("num", GangArgType.NUM),
    ("dim", GangArgType.DIM),
    ("static", GangArgType.STATIC),
)
"""`(keyword, GangArgType)` pairs in upstream's parse-attempt order."""

_GANG_KEYWORD_BY_TYPE = {ty: kw for kw, ty in _GANG_KEYWORDS}


def _parse_gang_value_group(
    parser: Parser,
) -> tuple[
    tuple[UnresolvedOperand, ...],
    tuple[Attribute, ...],
    tuple[GangArgTypeAttr, ...],
    DeviceTypeAttr,
]:
    """Parse `{ kw=%v:T (`,` kw=%v:T)* }` plus an optional `[#dt]` suffix.

    Returns `(operands, types, gang_arg_types, device_type)`. Mirrors the
    inner loop of upstream's `parseGangClause`, which retries each
    `num=` / `dim=` / `static=` keyword in order; raises on an empty group
    matching upstream's "expect at least one of num, dim or static values"
    diagnostic.
    """
    with parser.in_braces():
        operands: list[UnresolvedOperand] = []
        types: list[Attribute] = []
        arg_types: list[GangArgTypeAttr] = []
        need_comma = False
        while True:
            if need_comma and parser.parse_optional_punctuation(",") is None:
                break
            matched = False
            for keyword, gang_arg_ty in _GANG_KEYWORDS:
                if parser.parse_optional_keyword(keyword) is None:
                    continue
                parser.parse_punctuation("=")
                operand, ty = _parse_typed_operand(parser)
                operands.append(operand)
                types.append(ty)
                arg_types.append(GangArgTypeAttr(gang_arg_ty))
                matched = True
                need_comma = True
                break
            if not matched:
                if need_comma:
                    parser.raise_error("new value expected after comma")
                break
        if not operands:
            parser.raise_error("expect at least one of num, dim or static values")
    dt = _parse_optional_device_type_suffix(parser)
    return tuple(operands), tuple(types), tuple(arg_types), dt


@irdl_custom_directive
class GangClause(CustomDirective):
    """Port of upstream `custom<GangClause>`.

    Follows a bare `gang` keyword in `acc.loop`'s format. The directive
    owns the optional surrounding parentheses. Syntax options after the
    keyword:
      bare                                     → `gang_only` = [#none]
      `(` `[` dts `]` `)`                      → keyword-only DT list, no operands
      `(` group (`,` group)* `)`               → gang operand groups
      `(` `[` dts `]` `,` group (`,` group)* `)` → mix of keyword-only and groups
    where each group is `{ kw=%v:T (`,` kw=%v:T)* }` (`[` dt `]`)? and
    `kw` ∈ {`num`, `dim`, `static`}.
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    gang_arg_types: AttributeVariable
    device_types: AttributeVariable
    segments: AttributeVariable
    gang_only: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return (
            bool(self.operands.get(op))
            or self.gang_only.get(op) is not None
            or self.device_types.get(op) is not None
        )

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if not parser.parse_optional_punctuation("("):
            self.gang_only.set(state, ArrayAttr([DeviceTypeAttr(DeviceType.NONE)]))
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        kw_only = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, lambda: _parse_device_type_attr(parser)
        )
        if kw_only is not None:
            self.gang_only.set(state, ArrayAttr(kw_only))

        if parser.parse_optional_punctuation(")"):
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        if kw_only is not None:
            parser.parse_punctuation(",")

        groups = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, lambda: _parse_gang_value_group(parser)
        )
        parser.parse_punctuation(")")

        all_operands: list[UnresolvedOperand] = []
        all_types: list[Attribute] = []
        arg_types: list[GangArgTypeAttr] = []
        dts: list[DeviceTypeAttr] = []
        segs: list[int] = []
        for group_operands, group_types, group_arg_types, dt in groups:
            all_operands.extend(group_operands)
            all_types.extend(group_types)
            arg_types.extend(group_arg_types)
            dts.append(dt)
            segs.append(len(group_operands))

        self.operands.set(state, tuple(all_operands))
        self.operand_types.set(state, tuple(all_types))
        self.gang_arg_types.set(state, ArrayAttr(arg_types))
        self.device_types.set(state, ArrayAttr(dts))
        self.segments.set(state, DenseArrayBase.from_list(i32, segs))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        gang_only = self.gang_only.get(op)

        if not operands and gang_only == _DEVICE_TYPE_ONLY_NONE:
            return

        printer.print_string("(")
        if (
            isa(gang_only, ArrayAttr)
            and gang_only.data
            and gang_only != _DEVICE_TYPE_ONLY_NONE
        ):
            printer.print_string("[")
            printer.print_list(gang_only.data, printer.print_attribute)
            printer.print_string("]")
            if operands:
                printer.print_string(", ")

        if operands:
            dts = (
                attr.data
                if isa(attr := self.device_types.get(op), ArrayAttr)
                else (DeviceTypeAttr(DeviceType.NONE),)
            )
            seg_values: Sequence[int] = (
                segments.get_values()
                if isinstance(segments := self.segments.get(op), DenseArrayBase)
                else (len(operands),)
            )
            arg_type_values: Sequence[Attribute] = (
                arg_attrs.data
                if isa(arg_attrs := self.gang_arg_types.get(op), ArrayAttr)
                else ()
            )
            idx = 0
            for group_idx, (size, dt) in enumerate(zip(seg_values, dts, strict=True)):
                if group_idx:
                    printer.print_string(", ")
                printer.print_string("{")
                for i in range(size):
                    if i:
                        printer.print_string(", ")
                    arg_ty_attr = (
                        arg_type_values[idx]
                        if idx < len(arg_type_values)
                        else GangArgTypeAttr(GangArgType.NUM)
                    )
                    keyword = (
                        _GANG_KEYWORD_BY_TYPE[arg_ty_attr.data]
                        if isinstance(arg_ty_attr, GangArgTypeAttr)
                        else "num"
                    )
                    printer.print_string(keyword)
                    printer.print_string("=")
                    _print_typed_operand(printer, operands[idx])
                    idx += 1
                printer.print_string("}")
                _print_device_type_suffix(printer, dt)
        printer.print_string(")")
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class LoopControl(CustomDirective):
    """Port of upstream `custom<LoopControl>`.

    Owns `acc.loop`'s region and the optional `control(...) = (...) to (...)
    step (...)` header. Syntax:
      `control` `(` arg `:` T (`,` arg `:` T)* `)` `=`
        `(` lb (`,` lb)* `:` T (`,` T)* `)` `to`
        `(` ub (`,` ub)* `:` T (`,` T)* `)` `step`
        `(` st (`,` st)* `:` T (`,` T)* `)`
        region
      | region

    The first form provides induction variables to the region's entry
    block (matching upstream's `parseRegion(region, inductionVars)`); the
    second form is the container-like loop, which has no induction
    variables.
    """

    region: RegionVariable
    lowerbound: VariadicOperandVariable
    lowerbound_types: TypeDirective
    upperbound: VariadicOperandVariable
    upperbound_types: TypeDirective
    step: VariadicOperandVariable
    step_types: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if parser.parse_optional_keyword("control") is None:
            self.lowerbound.set(state, ())
            self.lowerbound_types.set(state, ())
            self.upperbound.set(state, ())
            self.upperbound_types.set(state, ())
            self.step.set(state, ())
            self.step_types.set(state, ())
            self.region.set(state, parser.parse_region())
            return True

        with parser.in_parens():
            args = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parser.parse_argument
            )
        parser.parse_punctuation("=")
        lb_ops, lb_types = _parse_loop_bound_group(parser, len(args))
        parser.parse_keyword("to")
        ub_ops, ub_types = _parse_loop_bound_group(parser, len(args))
        parser.parse_keyword("step")
        st_ops, st_types = _parse_loop_bound_group(parser, len(args))

        self.lowerbound.set(state, lb_ops)
        self.lowerbound_types.set(state, lb_types)
        self.upperbound.set(state, ub_ops)
        self.upperbound_types.set(state, ub_types)
        self.step.set(state, st_ops)
        self.step_types.set(state, st_types)
        self.region.set(state, parser.parse_region(args))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        region = self.region.get(op)
        entry_args: Sequence[SSAValue] = region.block.args if region.blocks else ()
        lb_ops = self.lowerbound.get(op)
        ub_ops = self.upperbound.get(op)
        st_ops = self.step.get(op)

        state.print_whitespace(printer)
        if entry_args:
            printer.print_string("control(")
            printer.print_list(
                entry_args,
                lambda arg: _print_typed_operand(printer, arg),
            )
            printer.print_string(") = (")
            _print_loop_bound_group(printer, lb_ops)
            printer.print_string(") to (")
            _print_loop_bound_group(printer, ub_ops)
            printer.print_string(") step (")
            _print_loop_bound_group(printer, st_ops)
            printer.print_string(") ")

        printer.print_region(region, print_entry_block_args=False)
        state.should_emit_space = True
        state.last_was_punctuation = False


def _parse_loop_bound_group(
    parser: Parser, count: int
) -> tuple[tuple[UnresolvedOperand, ...], tuple[Attribute, ...]]:
    """Parse `(%a (`,` %a)* `:` T (`,` T)*)` with exactly `count` operands and types."""
    with parser.in_parens():
        operands = tuple(
            parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parser.parse_unresolved_operand
            )
        )
        if len(operands) != count:
            parser.raise_error(f"expected {count} operands")
        parser.parse_punctuation(":")
        types = tuple(
            parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_type)
        )
        if len(types) != count:
            parser.raise_error(f"expected {count} types")
    return operands, types


def _print_loop_bound_group(printer: Printer, operands: Sequence[SSAValue]) -> None:
    """Print `%a (`,` %a)* `:` T (`,` T)*` for a loop control bound group."""
    printer.print_list(operands, printer.print_ssa_value)
    printer.print_string(" : ")
    printer.print_list((operand.type for operand in operands), printer.print_attribute)


@irdl_custom_directive
class BindName(CustomDirective):
    """Port of upstream `custom<BindName>($bindIdName, $bindStrName,
    $bindIdNameDeviceType, $bindStrNameDeviceType)`.

    Body of `acc.routine`'s `bind(...)` clause. Each entry is either a
    SymbolRefAttr (`@name`) or a StringAttr (`"name"`) optionally followed by
    `[#acc.device_type<...>]`. SymbolRef entries land in `bindIdName` (paired
    with `bindIdNameDeviceType`); String entries land in `bindStrName` (paired
    with `bindStrNameDeviceType`). On print, all id-name entries are emitted
    first, then all str-name entries — mirroring upstream's two-pass print.
    """

    bind_id_name: AttributeVariable
    bind_str_name: AttributeVariable
    bind_id_name_device_type: AttributeVariable
    bind_str_name_device_type: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return (
            self.bind_id_name.get(op) is not None
            or self.bind_str_name.get(op) is not None
        )

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        id_names: list[SymbolRefAttr] = []
        str_names: list[StringAttr] = []
        id_dts: list[DeviceTypeAttr] = []
        str_dts: list[DeviceTypeAttr] = []

        def parse_one() -> None:
            attr = parser.parse_attribute()
            dt = _parse_optional_device_type_suffix(parser)
            if isinstance(attr, SymbolRefAttr):
                id_names.append(attr)
                id_dts.append(dt)
            elif isinstance(attr, StringAttr):
                str_names.append(attr)
                str_dts.append(dt)
            else:
                parser.raise_error(
                    "expected SymbolRef or string attribute in bind clause"
                )

        parser.parse_comma_separated_list(parser.Delimiter.NONE, parse_one)

        if id_names:
            self.bind_id_name.set(state, ArrayAttr(id_names))
            self.bind_id_name_device_type.set(state, ArrayAttr(id_dts))
        if str_names:
            self.bind_str_name.set(state, ArrayAttr(str_names))
            self.bind_str_name_device_type.set(state, ArrayAttr(str_dts))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        # Concatenate the (id-name, dt) and (str-name, dt) sequences in that
        # order — upstream's two-pass print emits all SymbolRef entries
        # before all String entries regardless of source order.
        entries: list[tuple[Attribute, Attribute]] = []
        for names_attr, dts_var in (
            (self.bind_id_name.get(op), self.bind_id_name_device_type),
            (self.bind_str_name.get(op), self.bind_str_name_device_type),
        ):
            if isa(names_attr, ArrayAttr) and names_attr.data:
                dts = (
                    attr.data
                    if isa(attr := dts_var.get(op), ArrayAttr)
                    else (DeviceTypeAttr(DeviceType.NONE),) * len(names_attr.data)
                )
                entries.extend(zip(names_attr.data, dts, strict=True))

        def print_entry(pair: tuple[Attribute, Attribute]) -> None:
            printer.print_attribute(pair[0])
            _print_device_type_suffix(printer, pair[1])

        printer.print_list(entries, print_entry)
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class RoutineGangClause(CustomDirective):
    """Port of upstream `custom<RoutineGangClause>($gang, $gangDim,
    $gangDimDeviceType)`.

    Follows a bare `gang` keyword in `acc.routine`'s format. The directive
    owns the optional surrounding parentheses. Syntax options after the
    keyword:
      bare                                                → gang = [#none]
      `(` `[` dts `]` `)`                                 → keyword-only DT list
      `(` `dim` `:` <attr> ([dt])? (`,` ...)* `)`         → dim entries
      `(` `[` dts `]` `,` `dim` `:` <attr> ([dt])? `)`    → mix of both
    `dim:` parses an attribute (typically `1 : i64`), not an SSA operand.
    """

    gang: AttributeVariable
    gang_dim: AttributeVariable
    gang_dim_device_type: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        return self.gang.get(op) is not None or self.gang_dim.get(op) is not None

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if not parser.parse_optional_punctuation("("):
            self.gang.set(state, ArrayAttr([DeviceTypeAttr(DeviceType.NONE)]))
            return True

        kw_only = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, lambda: _parse_device_type_attr(parser)
        )
        if kw_only is not None:
            self.gang.set(state, ArrayAttr(kw_only))
            if parser.parse_optional_punctuation(")"):
                return True
            parser.parse_punctuation(",")

        gang_dim_attrs: list[Attribute] = []
        gang_dim_dt_attrs: list[DeviceTypeAttr] = []

        def parse_dim_entry() -> None:
            parser.parse_keyword("dim")
            parser.parse_punctuation(":")
            gang_dim_attrs.append(parser.parse_attribute())
            gang_dim_dt_attrs.append(_parse_optional_device_type_suffix(parser))

        parser.parse_comma_separated_list(parser.Delimiter.NONE, parse_dim_entry)
        parser.parse_punctuation(")")

        # `parse_comma_separated_list(Delimiter.NONE, ...)` requires at least
        # one element, so `gang_dim_attrs` is always non-empty here.
        self.gang_dim.set(state, ArrayAttr(gang_dim_attrs))
        self.gang_dim_device_type.set(state, ArrayAttr(gang_dim_dt_attrs))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        gang = self.gang.get(op)
        gang_dim_data = (
            attr.data if isa(attr := self.gang_dim.get(op), ArrayAttr) else ()
        )
        gang_dim_dts = (
            attr.data
            if isa(attr := self.gang_dim_device_type.get(op), ArrayAttr)
            else (DeviceTypeAttr(DeviceType.NONE),) * len(gang_dim_data)
        )

        # Mirror upstream's bare-keyword elision: gang == [#none] and no dim
        # entries → emit nothing past the parent's `gang` keyword.
        if not gang_dim_data and gang == _DEVICE_TYPE_ONLY_NONE:
            return

        printer.print_string("(")
        gang_data = gang.data if isa(gang, ArrayAttr) else ()
        if gang_data:
            printer.print_string("[")
            printer.print_list(gang_data, printer.print_attribute)
            printer.print_string("]")

        if gang_data and gang_dim_data:
            printer.print_string(", ")

        def print_dim(pair: tuple[Attribute, Attribute]) -> None:
            printer.print_string("dim: ")
            printer.print_attribute(pair[0])
            _print_device_type_suffix(printer, pair[1])

        printer.print_list(zip(gang_dim_data, gang_dim_dts, strict=True), print_dim)

        printer.print_string(")")
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class DeviceTypeArrayClause(CustomDirective):
    """Port of upstream `custom<DeviceTypeArrayAttr>($deviceTypes)`.

    Follows a bare keyword (e.g. `worker`) in `acc.routine`'s format. The
    directive owns the optional surrounding parentheses. Syntax options
    after the keyword:
      bare                  → device_types = [#none]
      `(` `[` dts `]` `)`   → list of device types
    Upstream's printer elides both the bare-`#none` form and any empty list,
    so an empty array round-trips as the bare keyword.
    """

    device_types: AttributeVariable

    def is_anchorable(self) -> bool:
        return True

    def is_present(self, op: IRDLOperation) -> bool:
        # Filter out the empty-array case (e.g. `worker = []` from generic
        # form, or upstream's printer eliding an unset slot to `[]`) so the
        # parent `worker` keyword is dropped on print, matching upstream's
        # `hasDeviceTypeValues`-gated emission.
        return isa(attr := self.device_types.get(op), ArrayAttr) and bool(attr.data)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if not parser.parse_optional_punctuation("("):
            self.device_types.set(state, ArrayAttr([DeviceTypeAttr(DeviceType.NONE)]))
            return True

        attrs = parser.parse_optional_comma_separated_list(
            parser.Delimiter.SQUARE, lambda: _parse_device_type_attr(parser)
        )
        parser.parse_punctuation(")")
        self.device_types.set(
            state, ArrayAttr(attrs) if attrs is not None else ArrayAttr(())
        )
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        # `is_present` guarantees a non-empty ArrayAttr.
        attr = self.device_types.get(op)
        assert isa(attr, ArrayAttr)
        assert attr.data
        if attr == _DEVICE_TYPE_ONLY_NONE:
            return
        printer.print_string("([")
        printer.print_list(attr.data, printer.print_attribute)
        printer.print_string("])")
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_op_definition
class ParallelOp(IRDLOperation):
    """
    Implementation of upstream acc.parallel.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accparallel-accparallelop).
    """

    name = "acc.parallel"

    async_operands = var_operand_def(IntegerType | IndexType)
    wait_operands = var_operand_def(IntegerType | IndexType)
    num_gangs = var_operand_def(IntegerType | IndexType)
    num_workers = var_operand_def(IntegerType | IndexType)
    vector_length = var_operand_def(IntegerType | IndexType)
    if_cond = opt_operand_def(I1)
    self_cond = opt_operand_def(I1)
    reduction_operands = var_operand_def()
    private_operands = var_operand_def()
    firstprivate_operands = var_operand_def()
    data_clause_operands = var_operand_def()

    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    wait_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="waitOperandsDeviceType"
    )
    wait_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="waitOperandsSegments"
    )
    has_wait_devnum = opt_prop_def(ArrayAttr[BoolAttr], prop_name="hasWaitDevnum")
    wait_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="waitOnly")
    num_gangs_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="numGangsSegments"
    )
    num_gangs_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="numGangsDeviceType"
    )
    num_workers_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="numWorkersDeviceType"
    )
    vector_length_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="vectorLengthDeviceType"
    )
    self_attr = opt_prop_def(UnitAttr, prop_name="selfAttr")
    default_attr = opt_prop_def(ClauseDefaultValueAttr, prop_name="defaultAttr")
    combined = opt_prop_def(UnitAttr)

    region = region_def("single_block")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (
        DeviceTypeOperands,
        NumGangs,
        DeviceTypeOperandsWithKeywordOnly,
        WaitClause,
    )

    assembly_format = (
        "(`combined` `(` `loop` `)` $combined^)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " (`async` custom<DeviceTypeOperandsWithKeywordOnly>($async_operands,"
        " type($async_operands), $asyncOperandsDeviceType, $asyncOnly)^)?"
        " (`firstprivate` `(` $firstprivate_operands^ `:`"
        " type($firstprivate_operands) `)`)?"
        " (`num_gangs` `(` custom<NumGangs>($num_gangs, type($num_gangs),"
        " $numGangsDeviceType, $numGangsSegments)^ `)`)?"
        " (`num_workers` `(` custom<DeviceTypeOperands>($num_workers,"
        " type($num_workers), $numWorkersDeviceType)^ `)`)?"
        " (`private` `(` $private_operands^ `:`"
        " type($private_operands) `)`)?"
        " (`vector_length` `(` custom<DeviceTypeOperands>($vector_length,"
        " type($vector_length), $vectorLengthDeviceType)^ `)`)?"
        " (`wait` custom<WaitClause>($wait_operands, type($wait_operands),"
        " $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,"
        " $waitOnly)^)?"
        " (`self` `(` $self_cond^ `)`)?"
        " (`if` `(` $if_cond^ `)`)?"
        " (`reduction` `(` $reduction_operands^ `:`"
        " type($reduction_operands) `)`)?"
        " $region attr-dict-with-keyword"
    )

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(YieldOp),
            RecursiveMemoryEffect(),
        )
    )

    def __init__(
        self,
        *,
        region: Region,
        async_operands: Sequence[SSAValue | Operation] = (),
        wait_operands: Sequence[SSAValue | Operation] = (),
        num_gangs: Sequence[SSAValue | Operation] = (),
        num_workers: Sequence[SSAValue | Operation] = (),
        vector_length: Sequence[SSAValue | Operation] = (),
        if_cond: SSAValue | Operation | None = None,
        self_cond: SSAValue | Operation | None = None,
        reduction_operands: Sequence[SSAValue | Operation] = (),
        private_operands: Sequence[SSAValue | Operation] = (),
        firstprivate_operands: Sequence[SSAValue | Operation] = (),
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_segments: DenseArrayBase | None = None,
        has_wait_devnum: ArrayAttr[BoolAttr] | None = None,
        wait_only: ArrayAttr[DeviceTypeAttr] | None = None,
        num_gangs_segments: DenseArrayBase | None = None,
        num_gangs_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        num_workers_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        vector_length_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        self_attr: UnitAttr | bool = False,
        default_attr: ClauseDefaultValueAttr | ClauseDefaultValue | None = None,
        combined: UnitAttr | bool = False,
    ) -> None:
        self_attr_prop: UnitAttr | None = (
            (UnitAttr() if self_attr else None)
            if isinstance(self_attr, bool)
            else self_attr
        )
        combined_prop: UnitAttr | None = (
            (UnitAttr() if combined else None)
            if isinstance(combined, bool)
            else combined
        )
        default_prop: ClauseDefaultValueAttr | None = (
            ClauseDefaultValueAttr(default_attr)
            if isinstance(default_attr, ClauseDefaultValue)
            else default_attr
        )
        super().__init__(
            operands=[
                async_operands,
                wait_operands,
                num_gangs,
                num_workers,
                vector_length,
                [if_cond] if if_cond is not None else [],
                [self_cond] if self_cond is not None else [],
                reduction_operands,
                private_operands,
                firstprivate_operands,
                data_clause_operands,
            ],
            properties={
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "waitOperandsDeviceType": wait_operands_device_type,
                "waitOperandsSegments": wait_operands_segments,
                "hasWaitDevnum": has_wait_devnum,
                "waitOnly": wait_only,
                "numGangsSegments": num_gangs_segments,
                "numGangsDeviceType": num_gangs_device_type,
                "numWorkersDeviceType": num_workers_device_type,
                "vectorLengthDeviceType": vector_length_device_type,
                "selfAttr": self_attr_prop,
                "defaultAttr": default_prop,
                "combined": combined_prop,
            },
            regions=[region],
        )


@irdl_op_definition
class SerialOp(IRDLOperation):
    """
    Implementation of upstream acc.serial.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accserial-accserialop).
    """

    name = "acc.serial"

    async_operands = var_operand_def(IntegerType | IndexType)
    wait_operands = var_operand_def(IntegerType | IndexType)
    if_cond = opt_operand_def(I1)
    self_cond = opt_operand_def(I1)
    reduction_operands = var_operand_def()
    private_operands = var_operand_def()
    firstprivate_operands = var_operand_def()
    data_clause_operands = var_operand_def()

    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    wait_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="waitOperandsDeviceType"
    )
    wait_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="waitOperandsSegments"
    )
    has_wait_devnum = opt_prop_def(ArrayAttr[BoolAttr], prop_name="hasWaitDevnum")
    wait_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="waitOnly")
    self_attr = opt_prop_def(UnitAttr, prop_name="selfAttr")
    default_attr = opt_prop_def(ClauseDefaultValueAttr, prop_name="defaultAttr")
    combined = opt_prop_def(UnitAttr)

    region = region_def("single_block")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (
        DeviceTypeOperandsWithKeywordOnly,
        WaitClause,
    )

    assembly_format = (
        "(`combined` `(` `loop` `)` $combined^)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " (`async` custom<DeviceTypeOperandsWithKeywordOnly>($async_operands,"
        " type($async_operands), $asyncOperandsDeviceType, $asyncOnly)^)?"
        " (`firstprivate` `(` $firstprivate_operands^ `:`"
        " type($firstprivate_operands) `)`)?"
        " (`private` `(` $private_operands^ `:`"
        " type($private_operands) `)`)?"
        " (`wait` custom<WaitClause>($wait_operands, type($wait_operands),"
        " $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,"
        " $waitOnly)^)?"
        " (`self` `(` $self_cond^ `)`)?"
        " (`if` `(` $if_cond^ `)`)?"
        " (`reduction` `(` $reduction_operands^ `:`"
        " type($reduction_operands) `)`)?"
        " $region attr-dict-with-keyword"
    )

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(YieldOp),
            RecursiveMemoryEffect(),
        )
    )

    def __init__(
        self,
        *,
        region: Region,
        async_operands: Sequence[SSAValue | Operation] = (),
        wait_operands: Sequence[SSAValue | Operation] = (),
        if_cond: SSAValue | Operation | None = None,
        self_cond: SSAValue | Operation | None = None,
        reduction_operands: Sequence[SSAValue | Operation] = (),
        private_operands: Sequence[SSAValue | Operation] = (),
        firstprivate_operands: Sequence[SSAValue | Operation] = (),
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_segments: DenseArrayBase | None = None,
        has_wait_devnum: ArrayAttr[BoolAttr] | None = None,
        wait_only: ArrayAttr[DeviceTypeAttr] | None = None,
        self_attr: UnitAttr | bool = False,
        default_attr: ClauseDefaultValueAttr | ClauseDefaultValue | None = None,
        combined: UnitAttr | bool = False,
    ) -> None:
        self_attr_prop: UnitAttr | None = (
            (UnitAttr() if self_attr else None)
            if isinstance(self_attr, bool)
            else self_attr
        )
        combined_prop: UnitAttr | None = (
            (UnitAttr() if combined else None)
            if isinstance(combined, bool)
            else combined
        )
        default_prop: ClauseDefaultValueAttr | None = (
            ClauseDefaultValueAttr(default_attr)
            if isinstance(default_attr, ClauseDefaultValue)
            else default_attr
        )
        super().__init__(
            operands=[
                async_operands,
                wait_operands,
                [if_cond] if if_cond is not None else [],
                [self_cond] if self_cond is not None else [],
                reduction_operands,
                private_operands,
                firstprivate_operands,
                data_clause_operands,
            ],
            properties={
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "waitOperandsDeviceType": wait_operands_device_type,
                "waitOperandsSegments": wait_operands_segments,
                "hasWaitDevnum": has_wait_devnum,
                "waitOnly": wait_only,
                "selfAttr": self_attr_prop,
                "defaultAttr": default_prop,
                "combined": combined_prop,
            },
            regions=[region],
        )


@irdl_op_definition
class KernelsOp(IRDLOperation):
    """
    Implementation of upstream acc.kernels.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#acckernels-acckernelsop).
    """

    name = "acc.kernels"

    async_operands = var_operand_def(IntegerType | IndexType)
    wait_operands = var_operand_def(IntegerType | IndexType)
    num_gangs = var_operand_def(IntegerType | IndexType)
    num_workers = var_operand_def(IntegerType | IndexType)
    vector_length = var_operand_def(IntegerType | IndexType)
    if_cond = opt_operand_def(I1)
    self_cond = opt_operand_def(I1)
    reduction_operands = var_operand_def()
    private_operands = var_operand_def()
    firstprivate_operands = var_operand_def()
    data_clause_operands = var_operand_def()

    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    wait_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="waitOperandsDeviceType"
    )
    wait_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="waitOperandsSegments"
    )
    has_wait_devnum = opt_prop_def(ArrayAttr[BoolAttr], prop_name="hasWaitDevnum")
    wait_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="waitOnly")
    num_gangs_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="numGangsSegments"
    )
    num_gangs_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="numGangsDeviceType"
    )
    num_workers_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="numWorkersDeviceType"
    )
    vector_length_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="vectorLengthDeviceType"
    )
    self_attr = opt_prop_def(UnitAttr, prop_name="selfAttr")
    default_attr = opt_prop_def(ClauseDefaultValueAttr, prop_name="defaultAttr")
    combined = opt_prop_def(UnitAttr)

    region = region_def()

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (
        DeviceTypeOperands,
        NumGangs,
        DeviceTypeOperandsWithKeywordOnly,
        WaitClause,
    )

    assembly_format = (
        "(`combined` `(` `loop` `)` $combined^)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " (`async` custom<DeviceTypeOperandsWithKeywordOnly>($async_operands,"
        " type($async_operands), $asyncOperandsDeviceType, $asyncOnly)^)?"
        " (`firstprivate` `(` $firstprivate_operands^ `:`"
        " type($firstprivate_operands) `)`)?"
        " (`num_gangs` `(` custom<NumGangs>($num_gangs, type($num_gangs),"
        " $numGangsDeviceType, $numGangsSegments)^ `)`)?"
        " (`num_workers` `(` custom<DeviceTypeOperands>($num_workers,"
        " type($num_workers), $numWorkersDeviceType)^ `)`)?"
        " (`private` `(` $private_operands^ `:`"
        " type($private_operands) `)`)?"
        " (`vector_length` `(` custom<DeviceTypeOperands>($vector_length,"
        " type($vector_length), $vectorLengthDeviceType)^ `)`)?"
        " (`wait` custom<WaitClause>($wait_operands, type($wait_operands),"
        " $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,"
        " $waitOnly)^)?"
        " (`self` `(` $self_cond^ `)`)?"
        " (`if` `(` $if_cond^ `)`)?"
        " (`reduction` `(` $reduction_operands^ `:`"
        " type($reduction_operands) `)`)?"
        " $region attr-dict-with-keyword"
    )

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(TerminatorOp),
            RecursiveMemoryEffect(),
        )
    )

    def __init__(
        self,
        *,
        region: Region,
        async_operands: Sequence[SSAValue | Operation] = (),
        wait_operands: Sequence[SSAValue | Operation] = (),
        num_gangs: Sequence[SSAValue | Operation] = (),
        num_workers: Sequence[SSAValue | Operation] = (),
        vector_length: Sequence[SSAValue | Operation] = (),
        if_cond: SSAValue | Operation | None = None,
        self_cond: SSAValue | Operation | None = None,
        reduction_operands: Sequence[SSAValue | Operation] = (),
        private_operands: Sequence[SSAValue | Operation] = (),
        firstprivate_operands: Sequence[SSAValue | Operation] = (),
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_segments: DenseArrayBase | None = None,
        has_wait_devnum: ArrayAttr[BoolAttr] | None = None,
        wait_only: ArrayAttr[DeviceTypeAttr] | None = None,
        num_gangs_segments: DenseArrayBase | None = None,
        num_gangs_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        num_workers_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        vector_length_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        self_attr: UnitAttr | bool = False,
        default_attr: ClauseDefaultValueAttr | ClauseDefaultValue | None = None,
        combined: UnitAttr | bool = False,
    ) -> None:
        self_attr_prop: UnitAttr | None = (
            (UnitAttr() if self_attr else None)
            if isinstance(self_attr, bool)
            else self_attr
        )
        combined_prop: UnitAttr | None = (
            (UnitAttr() if combined else None)
            if isinstance(combined, bool)
            else combined
        )
        default_prop: ClauseDefaultValueAttr | None = (
            ClauseDefaultValueAttr(default_attr)
            if isinstance(default_attr, ClauseDefaultValue)
            else default_attr
        )
        super().__init__(
            operands=[
                async_operands,
                wait_operands,
                num_gangs,
                num_workers,
                vector_length,
                [if_cond] if if_cond is not None else [],
                [self_cond] if self_cond is not None else [],
                reduction_operands,
                private_operands,
                firstprivate_operands,
                data_clause_operands,
            ],
            properties={
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "waitOperandsDeviceType": wait_operands_device_type,
                "waitOperandsSegments": wait_operands_segments,
                "hasWaitDevnum": has_wait_devnum,
                "waitOnly": wait_only,
                "numGangsSegments": num_gangs_segments,
                "numGangsDeviceType": num_gangs_device_type,
                "numWorkersDeviceType": num_workers_device_type,
                "vectorLengthDeviceType": vector_length_device_type,
                "selfAttr": self_attr_prop,
                "defaultAttr": default_prop,
                "combined": combined_prop,
            },
            regions=[region],
        )


@irdl_op_definition
class KernelEnvironmentOp(IRDLOperation):
    """
    Implementation of upstream acc.kernel_environment — a decomposition of an
    OpenACC compute construct that captures only the data-mapping and
    asynchronous-behavior clauses (data / async / wait), leaving kernel
    execution parallelism and privatization to be handled separately. The
    body is a single block with no terminator and typically wraps a
    `gpu.launch`.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#acckernel_environment-acckernelenvironmentop).
    """

    name = "acc.kernel_environment"

    data_clause_operands = var_operand_def()
    async_operands = var_operand_def(IntegerType | IndexType)
    wait_operands = var_operand_def(IntegerType | IndexType)

    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    wait_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="waitOperandsSegments"
    )
    wait_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="waitOperandsDeviceType"
    )
    has_wait_devnum = opt_prop_def(ArrayAttr[BoolAttr], prop_name="hasWaitDevnum")
    wait_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="waitOnly")

    region = region_def("single_block")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (KernelEnvironmentClauses,)

    # Port of upstream's `oilist(dataOperands | async | wait)`: the three
    # clause keywords are order-independent on parse, and the directive
    # always emits in upstream td-definition order (data, async, wait) on
    # print so the round-trip is bit-identical with mlir-opt.
    assembly_format = (
        "custom<KernelEnvironmentClauses>($data_clause_operands,"
        " type($data_clause_operands), $async_operands, type($async_operands),"
        " $asyncOperandsDeviceType, $asyncOnly, $wait_operands,"
        " type($wait_operands), $waitOperandsDeviceType, $waitOperandsSegments,"
        " $hasWaitDevnum, $waitOnly)"
        " $region attr-dict-with-keyword"
    )

    traits = traits_def(NoTerminator(), RecursiveMemoryEffect())

    def __init__(
        self,
        *,
        region: Region,
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        async_operands: Sequence[SSAValue | Operation] = (),
        wait_operands: Sequence[SSAValue | Operation] = (),
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_segments: DenseArrayBase | None = None,
        wait_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        has_wait_devnum: ArrayAttr[BoolAttr] | None = None,
        wait_only: ArrayAttr[DeviceTypeAttr] | None = None,
    ) -> None:
        super().__init__(
            operands=[
                data_clause_operands,
                async_operands,
                wait_operands,
            ],
            properties={
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "waitOperandsSegments": wait_operands_segments,
                "waitOperandsDeviceType": wait_operands_device_type,
                "hasWaitDevnum": has_wait_devnum,
                "waitOnly": wait_only,
            },
            regions=[region],
        )


@irdl_op_definition
class LoopOp(IRDLOperation):
    """
    Implementation of upstream acc.loop — the OpenACC loop construct.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accloop-accloopop).

    Carries the loop's induction variables (`lowerbound` / `upperbound` /
    `step`) plus per-device-type clauses for `gang` / `worker` / `vector`,
    `private` / `firstprivate` / `reduction` data ops, `tile` / `cache`
    operands, `collapse` group sizes, and `seq` / `independent` / `auto`
    parallelism mode markers. The body is an arbitrary region terminated
    by `acc.yield`.

    The container-like loop count check from upstream's verifier (which
    uses `LoopLikeOpInterface` to enumerate loop ops) is intentionally
    skipped here — xDSL has no equivalent interface yet, so PR 17 of the
    OpenACC roadmap will wire it up alongside the other interface-driven
    traits.
    """

    name = "acc.loop"

    lowerbound = var_operand_def(IntegerType | IndexType)
    upperbound = var_operand_def(IntegerType | IndexType)
    step = var_operand_def(IntegerType | IndexType)
    gang_operands = var_operand_def(IntegerType | IndexType)
    worker_num_operands = var_operand_def(IntegerType | IndexType)
    vector_operands = var_operand_def(IntegerType | IndexType)
    tile_operands = var_operand_def(IntegerType | IndexType)
    cache_operands = var_operand_def()
    private_operands = var_operand_def()
    firstprivate_operands = var_operand_def()
    reduction_operands = var_operand_def()

    inclusive_upperbound = opt_prop_def(
        DenseArrayBase.constr(IntegerType(1)), prop_name="inclusiveUpperbound"
    )
    collapse = opt_prop_def(ArrayAttr[IntegerAttr])
    collapse_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="collapseDeviceType"
    )
    gang_operands_arg_type = opt_prop_def(
        ArrayAttr[GangArgTypeAttr], prop_name="gangOperandsArgType"
    )
    gang_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="gangOperandsSegments"
    )
    gang_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="gangOperandsDeviceType"
    )
    worker_num_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="workerNumOperandsDeviceType"
    )
    vector_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="vectorOperandsDeviceType"
    )
    seq = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    independent = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    auto_ = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="auto_")
    gang = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    worker = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    vector = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    tile_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="tileOperandsSegments"
    )
    tile_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="tileOperandsDeviceType"
    )
    combined = opt_prop_def(CombinedConstructsTypeAttr)
    unstructured = opt_prop_def(UnitAttr)

    results_ = var_result_def()

    region = region_def()

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (
        CombinedConstructsLoop,
        GangClause,
        DeviceTypeOperandsWithKeywordOnly,
        DeviceTypeOperandsWithSegment,
        LoopControl,
    )

    assembly_format = (
        "(`combined` `(` custom<CombinedConstructsLoop>($combined)^ `)`)?"
        " (`gang` custom<GangClause>($gang_operands, type($gang_operands),"
        " $gangOperandsArgType, $gangOperandsDeviceType,"
        " $gangOperandsSegments, $gang)^)?"
        " (`worker` custom<DeviceTypeOperandsWithKeywordOnly>("
        "$worker_num_operands, type($worker_num_operands),"
        " $workerNumOperandsDeviceType, $worker)^)?"
        " (`vector` custom<DeviceTypeOperandsWithKeywordOnly>($vector_operands,"
        " type($vector_operands), $vectorOperandsDeviceType, $vector)^)?"
        " (`private` `(` $private_operands^ `:` type($private_operands) `)`)?"
        " (`firstprivate` `(` $firstprivate_operands^ `:`"
        " type($firstprivate_operands) `)`)?"
        " (`tile` `(` custom<DeviceTypeOperandsWithSegment>($tile_operands,"
        " type($tile_operands), $tileOperandsDeviceType,"
        " $tileOperandsSegments)^ `)`)?"
        " (`reduction` `(` $reduction_operands^ `:`"
        " type($reduction_operands) `)`)?"
        " (`cache` `(` $cache_operands^ `:` type($cache_operands) `)`)?"
        " custom<LoopControl>($region, $lowerbound, type($lowerbound),"
        " $upperbound, type($upperbound), $step, type($step))"
        " (`(` type($results_)^ `)`)?"
        " attr-dict-with-keyword"
    )

    traits = lazy_traits_def(lambda: (RecursiveMemoryEffect(),))

    def __init__(
        self,
        *,
        region: Region,
        lowerbound: Sequence[SSAValue | Operation] = (),
        upperbound: Sequence[SSAValue | Operation] = (),
        step: Sequence[SSAValue | Operation] = (),
        gang_operands: Sequence[SSAValue | Operation] = (),
        worker_num_operands: Sequence[SSAValue | Operation] = (),
        vector_operands: Sequence[SSAValue | Operation] = (),
        tile_operands: Sequence[SSAValue | Operation] = (),
        cache_operands: Sequence[SSAValue | Operation] = (),
        private_operands: Sequence[SSAValue | Operation] = (),
        firstprivate_operands: Sequence[SSAValue | Operation] = (),
        reduction_operands: Sequence[SSAValue | Operation] = (),
        result_types: Sequence[Attribute] = (),
        inclusive_upperbound: DenseArrayBase | None = None,
        collapse: ArrayAttr[IntegerAttr] | None = None,
        collapse_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        gang_operands_arg_type: ArrayAttr[GangArgTypeAttr] | None = None,
        gang_operands_segments: DenseArrayBase | None = None,
        gang_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        worker_num_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        vector_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        seq: ArrayAttr[DeviceTypeAttr] | None = None,
        independent: ArrayAttr[DeviceTypeAttr] | None = None,
        auto_: ArrayAttr[DeviceTypeAttr] | None = None,
        gang: ArrayAttr[DeviceTypeAttr] | None = None,
        worker: ArrayAttr[DeviceTypeAttr] | None = None,
        vector: ArrayAttr[DeviceTypeAttr] | None = None,
        tile_operands_segments: DenseArrayBase | None = None,
        tile_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        combined: CombinedConstructsTypeAttr | CombinedConstructsType | None = None,
        unstructured: UnitAttr | bool = False,
        par_mode: LoopParMode | None = None,
    ) -> None:
        unstructured_prop: UnitAttr | None = (
            (UnitAttr() if unstructured else None)
            if isinstance(unstructured, bool)
            else unstructured
        )
        combined_prop: CombinedConstructsTypeAttr | None = (
            CombinedConstructsTypeAttr(combined)
            if isinstance(combined, CombinedConstructsType)
            else combined
        )
        # Mirror upstream's `LoopParMode`-driven builder: a single device-`none`
        # entry on the matching seq/independent/auto array. Explicit
        # `seq=` / `independent=` / `auto_=` arguments win if also given.
        if par_mode is not None:
            if par_mode is LoopParMode.SEQ and seq is None:
                seq = _DEVICE_TYPE_ONLY_NONE
            elif par_mode is LoopParMode.INDEPENDENT and independent is None:
                independent = _DEVICE_TYPE_ONLY_NONE
            elif par_mode is LoopParMode.AUTO and auto_ is None:
                auto_ = _DEVICE_TYPE_ONLY_NONE

        super().__init__(
            operands=[
                lowerbound,
                upperbound,
                step,
                gang_operands,
                worker_num_operands,
                vector_operands,
                tile_operands,
                cache_operands,
                private_operands,
                firstprivate_operands,
                reduction_operands,
            ],
            properties={
                "inclusiveUpperbound": inclusive_upperbound,
                "collapse": collapse,
                "collapseDeviceType": collapse_device_type,
                "gangOperandsArgType": gang_operands_arg_type,
                "gangOperandsSegments": gang_operands_segments,
                "gangOperandsDeviceType": gang_operands_device_type,
                "workerNumOperandsDeviceType": worker_num_operands_device_type,
                "vectorOperandsDeviceType": vector_operands_device_type,
                "seq": seq,
                "independent": independent,
                "auto_": auto_,
                "gang": gang,
                "worker": worker,
                "vector": vector,
                "tileOperandsSegments": tile_operands_segments,
                "tileOperandsDeviceType": tile_operands_device_type,
                "combined": combined_prop,
                "unstructured": unstructured_prop,
            },
            regions=[region],
            result_types=[result_types],
        )

    def verify_(self) -> None:
        # Mirrors `acc::LoopOp::verify`. The container-like loop checks
        # (sibling loops / collapse-count satisfaction) require a
        # `LoopLikeOpInterface`-style enumeration that xDSL doesn't have
        # yet — those land in PR 17 of the OpenACC roadmap.
        if len(self.upperbound) != len(self.step):
            raise VerifyException(
                "number of upperbounds expected to be the same as number of steps"
            )
        if len(self.upperbound) != len(self.lowerbound):
            raise VerifyException(
                "number of upperbounds expected to be the same as number of lowerbounds"
            )
        if (
            self.upperbound
            and self.inclusive_upperbound is not None
            and len(self.inclusive_upperbound.get_values()) != len(self.upperbound)
        ):
            raise VerifyException(
                "inclusiveUpperbound size is expected to be the same as upperbound size"
            )

        if self.collapse is not None and self.collapse_device_type is None:
            raise VerifyException(
                "collapse device_type attr must be define when collapse attr is present"
            )
        if (
            self.collapse is not None
            and self.collapse_device_type is not None
            and len(self.collapse) != len(self.collapse_device_type)
        ):
            raise VerifyException(
                "collapse attribute count must match collapse device_type count"
            )
        if (
            self.collapse_device_type is not None
            and (
                dup := _first_duplicate(
                    dt.data for dt in self.collapse_device_type.data
                )
            )
            is not None
        ):
            raise VerifyException(
                f"duplicate device_type `{dup.value}` found in "
                "collapseDeviceType attribute"
            )

        if self.gang_operands:
            if self.gang_operands_arg_type is None:
                raise VerifyException(
                    "gangOperandsArgType attribute must be defined when gang "
                    "operands are present"
                )
            if len(self.gang_operands) != len(self.gang_operands_arg_type):
                raise VerifyException(
                    "gangOperandsArgType attribute count must match gangOperands count"
                )
        if (
            self.gang is not None
            and (dup := _first_duplicate(dt.data for dt in self.gang.data)) is not None
        ):
            raise VerifyException(
                f"duplicate device_type `{dup.value}` found in gang attribute"
            )
        _verify_dt_and_segment_count_match(
            self.gang_operands,
            self.gang_operands_segments,
            self.gang_operands_device_type,
            "gang",
        )

        if (
            self.worker is not None
            and (dup := _first_duplicate(dt.data for dt in self.worker.data))
            is not None
        ):
            raise VerifyException(
                f"duplicate device_type `{dup.value}` found in worker attribute"
            )
        if (
            self.worker_num_operands_device_type is not None
            and (
                dup := _first_duplicate(
                    dt.data for dt in self.worker_num_operands_device_type.data
                )
            )
            is not None
        ):
            raise VerifyException(
                f"duplicate device_type `{dup.value}` found in "
                "workerNumOperandsDeviceType attribute"
            )
        _verify_dt_count_match(
            self.worker_num_operands,
            self.worker_num_operands_device_type,
            "worker",
        )

        if (
            self.vector is not None
            and (dup := _first_duplicate(dt.data for dt in self.vector.data))
            is not None
        ):
            raise VerifyException(
                f"duplicate device_type `{dup.value}` found in vector attribute"
            )
        if (
            self.vector_operands_device_type is not None
            and (
                dup := _first_duplicate(
                    dt.data for dt in self.vector_operands_device_type.data
                )
            )
            is not None
        ):
            raise VerifyException(
                f"duplicate device_type `{dup.value}` found in "
                "vectorOperandsDeviceType attribute"
            )
        _verify_dt_count_match(
            self.vector_operands, self.vector_operands_device_type, "vector"
        )

        _verify_dt_and_segment_count_match(
            self.tile_operands,
            self.tile_operands_segments,
            self.tile_operands_device_type,
            "tile",
        )

        # auto / independent / seq must not specify the same device type more
        # than once across all three.
        seen_device_types: set[DeviceType] = set()
        for attr in (self.auto_, self.independent, self.seq):
            if attr is None:
                continue
            for dt in attr.data:
                if dt.data in seen_device_types:
                    raise VerifyException(
                        "only one of auto, independent, seq can be present at "
                        "the same time"
                    )
                seen_device_types.add(dt.data)

        # At least one of auto / independent / seq must apply to the
        # device-`none` (default) device type.
        has_default = any(
            attr is not None and any(dt.data is DeviceType.NONE for dt in attr.data)
            for attr in (self.seq, self.independent, self.auto_)
        )
        if not has_default:
            raise VerifyException(
                "at least one of auto, independent, seq must be present"
            )

        # `gang` / `worker` / `vector` cannot coexist with `seq` for the same
        # device type.
        if self.seq is not None:
            seq_device_types = {dt.data for dt in self.seq.data}
            for attr in (self.gang, self.worker, self.vector):
                if attr is None:
                    continue
                if any(dt.data in seq_device_types for dt in attr.data):
                    raise VerifyException(
                        "gang, worker or vector cannot appear with seq"
                    )

        if self.unstructured is not None and self.lowerbound:
            raise VerifyException(
                "unstructured acc.loop must not have induction variables"
            )


_T = TypeVar("_T", bound=Hashable)


def _first_duplicate(els: Iterable[_T]) -> _T | None:
    """Return the first duplicate `el` in `els`, `None` if there are no duplicates."""
    seen: set[_T] = set()
    for el in els:
        if el in seen:
            return el
        seen.add(el)


def _verify_dt_count_match(
    operands: Sequence[SSAValue],
    device_types: ArrayAttr[DeviceTypeAttr] | None,
    keyword: str,
) -> None:
    """Mirror of upstream's `verifyDeviceTypeCountMatch`."""
    if operands and (device_types is None or len(device_types) != len(operands)):
        raise VerifyException(
            f"{keyword} operands count must match {keyword} device_type count"
        )


def _verify_dt_and_segment_count_match(
    operands: Sequence[SSAValue],
    segments: DenseArrayBase[IntegerType] | None,
    device_types: ArrayAttr[DeviceTypeAttr] | None,
    keyword: str,
) -> None:
    """Mirror of upstream's `verifyDeviceTypeAndSegmentCountMatch`."""
    seg_values = segments.get_values() if segments is not None else ()
    num_in_segments = sum(seg_values)
    nb_segments = len(seg_values)
    if num_in_segments != len(operands) or (device_types is None and operands):
        raise VerifyException(
            f"{keyword} operand count does not match count in segments"
        )
    if device_types is not None and len(device_types) != nb_segments:
        raise VerifyException(
            f"{keyword} segment count does not match device_type count"
        )


@irdl_op_definition
class DataOp(IRDLOperation):
    """
    Implementation of upstream acc.data — the structured data construct.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accdata-accdataop).
    """

    name = "acc.data"

    if_cond = opt_operand_def(I1)
    async_operands = var_operand_def(IntegerType | IndexType)
    wait_operands = var_operand_def(IntegerType | IndexType)
    data_clause_operands = var_operand_def()

    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    wait_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="waitOperandsSegments"
    )
    wait_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="waitOperandsDeviceType"
    )
    has_wait_devnum = opt_prop_def(ArrayAttr[BoolAttr], prop_name="hasWaitDevnum")
    wait_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="waitOnly")
    default_attr = opt_prop_def(ClauseDefaultValueAttr, prop_name="defaultAttr")

    region = region_def()

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (
        DeviceTypeOperandsWithKeywordOnly,
        WaitClause,
    )

    assembly_format = (
        "(`if` `(` $if_cond^ `)`)?"
        " (`async` custom<DeviceTypeOperandsWithKeywordOnly>($async_operands,"
        " type($async_operands), $asyncOperandsDeviceType, $asyncOnly)^)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " (`wait` custom<WaitClause>($wait_operands, type($wait_operands),"
        " $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,"
        " $waitOnly)^)?"
        " $region attr-dict-with-keyword"
    )

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(TerminatorOp),
            RecursiveMemoryEffect(),
        )
    )

    def __init__(
        self,
        *,
        region: Region,
        if_cond: SSAValue | Operation | None = None,
        async_operands: Sequence[SSAValue | Operation] = (),
        wait_operands: Sequence[SSAValue | Operation] = (),
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_segments: DenseArrayBase | None = None,
        wait_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        has_wait_devnum: ArrayAttr[BoolAttr] | None = None,
        wait_only: ArrayAttr[DeviceTypeAttr] | None = None,
        default_attr: ClauseDefaultValueAttr | ClauseDefaultValue | None = None,
    ) -> None:
        default_prop: ClauseDefaultValueAttr | None = (
            ClauseDefaultValueAttr(default_attr)
            if isinstance(default_attr, ClauseDefaultValue)
            else default_attr
        )
        super().__init__(
            operands=[
                [if_cond] if if_cond is not None else [],
                async_operands,
                wait_operands,
                data_clause_operands,
            ],
            properties={
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "waitOperandsSegments": wait_operands_segments,
                "waitOperandsDeviceType": wait_operands_device_type,
                "hasWaitDevnum": has_wait_devnum,
                "waitOnly": wait_only,
                "defaultAttr": default_prop,
            },
            regions=[region],
        )

    def verify_(self) -> None:
        # Mirrors `acc::DataOp::verify`: 2.6.5 requires at least one of the
        # data clauses (copy/copyin/copyout/create/no_create/present/deviceptr/
        # attach) or the `default` attribute.
        if (
            not self.async_operands
            and not self.wait_operands
            and not self.data_clause_operands
            and self.if_cond is None
            and self.default_attr is None
        ):
            raise VerifyException(
                "at least one operand or the default attribute "
                "must appear on the data operation"
            )
        for operand in self.data_clause_operands:
            defining_op = operand.owner
            if not isinstance(
                defining_op,
                (
                    AttachOp,
                    CopyinOp,
                    CopyoutOp,
                    CreateOp,
                    DeleteOp,
                    DetachOp,
                    DevicePtrOp,
                    GetDevicePtrOp,
                    NoCreateOp,
                    PresentOp,
                ),
            ):
                raise VerifyException(
                    "expect data entry/exit operation or acc.getdeviceptr "
                    "as defining op"
                )


@irdl_op_definition
class HostDataOp(IRDLOperation):
    """
    Implementation of upstream acc.host_data.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#acchost_data-acchost_dataop).
    """

    name = "acc.host_data"

    if_cond = opt_operand_def(I1)
    data_clause_operands = var_operand_def()

    if_present = opt_prop_def(UnitAttr, prop_name="ifPresent")

    region = region_def()

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    assembly_format = (
        "(`if` `(` $if_cond^ `)`)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " $region attr-dict-with-keyword"
    )

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(TerminatorOp),
            RecursiveMemoryEffect(),
        )
    )

    def __init__(
        self,
        *,
        region: Region,
        if_cond: SSAValue | Operation | None = None,
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        if_present: UnitAttr | bool = False,
    ) -> None:
        if_present_prop: UnitAttr | None = (
            (UnitAttr() if if_present else None)
            if isinstance(if_present, bool)
            else if_present
        )
        super().__init__(
            operands=[
                [if_cond] if if_cond is not None else [],
                data_clause_operands,
            ],
            properties={"ifPresent": if_present_prop},
            regions=[region],
        )

    def verify_(self) -> None:
        # Mirrors `acc::HostDataOp::verify`: at least one operand and each
        # operand must be defined by an `acc.use_device`.
        if not self.data_clause_operands:
            raise VerifyException(
                "at least one operand must appear on the host_data operation"
            )
        for operand in self.data_clause_operands:
            if not isinstance(operand.owner, UseDeviceOp):
                raise VerifyException("expect data entry operation as defining op")


# ---------------------------------------------------------------------------
# Entry data-clause ops
# ---------------------------------------------------------------------------
#
# Upstream models the entry data-clause family (`copyin`, `create`, `present`,
# `nocreate`, `attach`, `deviceptr`, `use_device`, `cache`,
# `declare_device_resident`, `declare_link`) as a single `OpenACC_DataEntryOperation`
# class with one operand-and-attribute shape. The xDSL port mirrors that shape
# in the abstract `_DataEntryOperation` mixin below; concrete leaves inherit it and
# add only what differs per op (`name`, the per-op `dataClause` default, and
# eventually the `NoMemoryEffect` trait on `acc.cache`). The IRDL field
# descriptors are inherited — same pattern already used by
# `_DataBoundsAccessorOp`.


class _DataEntryOperation(IRDLOperation, ABC):
    """Shared shape for the OpenACC entry data-clause ops.

    Mirrors upstream's `OpenACC_DataEntryOperation` td class. Concrete leaves
    override `name` and supply their own `dataClause` default. Everything else
    — the four operand groups, the `varType` / async / structured / implicit
    / modifiers properties, the optional `name` / `recipe`, the `accVar`
    result, the assembly format, and the keyword-only `__init__` — lives
    here.
    """

    var = operand_def()
    var_ptr_ptr = opt_operand_def()
    bounds = var_operand_def(DataBoundsType)
    async_operands = var_operand_def(IntegerType | IndexType)

    var_type = prop_def(TypeAttribute, prop_name="varType")
    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    structured = opt_prop_def(
        BoolAttr,
        default_value=IntegerAttr.from_bool(True),
        prop_name="structured",
    )
    implicit = opt_prop_def(
        BoolAttr,
        default_value=IntegerAttr.from_bool(False),
        prop_name="implicit",
    )
    modifiers = opt_prop_def(
        DataClauseModifierAttr,
        default_value=DataClauseModifierAttr(frozenset[DataClauseModifier]()),
        prop_name="modifiers",
    )
    var_name = opt_prop_def(StringAttr, prop_name="name")
    recipe = opt_prop_def(SymbolRefAttr, prop_name="recipe")

    acc_var = result_def()

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (Var, DataEntryOilist)

    # The four optional clauses (varPtrPtr / bounds / async / recipe) are
    # consumed by `DataEntryOilist`, which mirrors upstream's
    # `oilist(...)` — accepting them in any order on parse and emitting
    # them in the canonical td-definition order on print. The
    # `recipe(@sym)` clause is the SymbolRefAttr inline spelling matching
    # upstream's `recipe` `(` ... `)`; mlir-opt may emit the four clauses
    # in any order depending on context, so the oilist directive accepts
    # them all and normalizes to canonical order on print.
    assembly_format = (
        "custom<Var>($var, type($var), $varType)"
        " custom<DataEntryOilist>($var_ptr_ptr, type($var_ptr_ptr), $bounds,"
        " $async_operands, type($async_operands), $asyncOperandsDeviceType,"
        " $asyncOnly, $recipe)"
        " `->` type($acc_var) attr-dict"
    )

    def __init__(
        self,
        *,
        var: SSAValue | Operation,
        var_ptr_ptr: SSAValue | Operation | None = None,
        bounds: Sequence[SSAValue | Operation] = (),
        async_operands: Sequence[SSAValue | Operation] = (),
        var_type: TypeAttribute | None = None,
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        data_clause: DataClauseAttr | DataClause | None = None,
        structured: BoolAttr | bool | None = None,
        implicit: BoolAttr | bool | None = None,
        modifiers: DataClauseModifierAttr | None = None,
        var_name: StringAttr | str | None = None,
        recipe: SymbolRefAttr | None = None,
        acc_var_type: Attribute | None = None,
    ) -> None:
        var_value = SSAValue.get(var)
        if var_type is None:
            var_type = cast(TypeAttribute, _default_var_type(var_value.type))
        if acc_var_type is None:
            acc_var_type = var_value.type

        structured_prop: BoolAttr | None = (
            IntegerAttr.from_bool(structured)
            if isinstance(structured, bool)
            else structured
        )
        implicit_prop: BoolAttr | None = (
            IntegerAttr.from_bool(implicit) if isinstance(implicit, bool) else implicit
        )
        data_clause_prop: DataClauseAttr | None = (
            DataClauseAttr(data_clause)
            if isinstance(data_clause, DataClause)
            else data_clause
        )
        var_name_prop: StringAttr | None = (
            StringAttr(var_name) if isinstance(var_name, str) else var_name
        )

        super().__init__(
            operands=[
                [var],
                [var_ptr_ptr] if var_ptr_ptr is not None else [],
                list(bounds),
                list(async_operands),
            ],
            properties={
                "varType": var_type,
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "dataClause": data_clause_prop,
                "structured": structured_prop,
                "implicit": implicit_prop,
                "modifiers": modifiers,
                "name": var_name_prop,
                "recipe": recipe,
            },
            result_types=[acc_var_type],
        )


@irdl_op_definition
class CopyinOp(_DataEntryOperation):
    """Implementation of upstream acc.copyin."""

    name = "acc.copyin"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_COPYIN),
        prop_name="dataClause",
    )


@irdl_op_definition
class CreateOp(_DataEntryOperation):
    """Implementation of upstream acc.create."""

    name = "acc.create"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_CREATE),
        prop_name="dataClause",
    )


@irdl_op_definition
class PresentOp(_DataEntryOperation):
    """Implementation of upstream acc.present."""

    name = "acc.present"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_PRESENT),
        prop_name="dataClause",
    )


@irdl_op_definition
class NoCreateOp(_DataEntryOperation):
    """Implementation of upstream acc.nocreate."""

    name = "acc.nocreate"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_NO_CREATE),
        prop_name="dataClause",
    )


@irdl_op_definition
class AttachOp(_DataEntryOperation):
    """Implementation of upstream acc.attach."""

    name = "acc.attach"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_ATTACH),
        prop_name="dataClause",
    )


@irdl_op_definition
class DevicePtrOp(_DataEntryOperation):
    """Implementation of upstream acc.deviceptr."""

    name = "acc.deviceptr"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_DEVICEPTR),
        prop_name="dataClause",
    )


@irdl_op_definition
class UseDeviceOp(_DataEntryOperation):
    """Implementation of upstream acc.use_device."""

    name = "acc.use_device"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_USE_DEVICE),
        prop_name="dataClause",
    )


@irdl_op_definition
class CacheOp(_DataEntryOperation):
    """Implementation of upstream acc.cache. Carries `NoMemoryEffect` per upstream."""

    name = "acc.cache"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_CACHE),
        prop_name="dataClause",
    )

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class DeclareDeviceResidentOp(_DataEntryOperation):
    """Implementation of upstream acc.declare_device_resident."""

    name = "acc.declare_device_resident"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_DECLARE_DEVICE_RESIDENT),
        prop_name="dataClause",
    )


@irdl_op_definition
class DeclareLinkOp(_DataEntryOperation):
    """Implementation of upstream acc.declare_link."""

    name = "acc.declare_link"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_DECLARE_LINK),
        prop_name="dataClause",
    )


@irdl_op_definition
class GetDevicePtrOp(_DataEntryOperation):
    """Implementation of upstream acc.getdeviceptr.

    Used to get the `accVar` for a host variable when a structured data-entry
    op is not available; the natural pair for the unstructured exit ops below.
    """

    name = "acc.getdeviceptr"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_GETDEVICEPTR),
        prop_name="dataClause",
    )


@irdl_op_definition
class UpdateDeviceOp(_DataEntryOperation):
    """Implementation of upstream acc.update_device."""

    name = "acc.update_device"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_UPDATE_DEVICE),
        prop_name="dataClause",
    )


# ---------------------------------------------------------------------------
# Privatization data-clause ops
# ---------------------------------------------------------------------------
#
# `acc.private`, `acc.firstprivate`, `acc.firstprivate_map`, and
# `acc.reduction` are all `OpenACC_DataEntryOp` instances upstream — same
# operands and properties as the entry data-clause ops above, with the
# leaves differing only in `name` and the per-op `dataClause` default. The
# associated reduction operator is not carried on `acc.reduction` itself;
# it lives on the `acc.reduction.recipe` referenced via the inherited
# `recipe` SymbolRefAttr property. Both `acc.firstprivate` and
# `acc.firstprivate_map` use `acc_firstprivate` as their `dataClause`
# default — `firstprivate_map` is the decomposed mapping half of the
# firstprivate clause and shares the user-level clause name.


@irdl_op_definition
class PrivateOp(_DataEntryOperation):
    """Implementation of upstream acc.private."""

    name = "acc.private"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_PRIVATE),
        prop_name="dataClause",
    )


@irdl_op_definition
class FirstprivateOp(_DataEntryOperation):
    """Implementation of upstream acc.firstprivate."""

    name = "acc.firstprivate"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_FIRSTPRIVATE),
        prop_name="dataClause",
    )


@irdl_op_definition
class FirstprivateMapOp(_DataEntryOperation):
    """Implementation of upstream acc.firstprivate_map.

    Used to decompose firstprivate semantics — represents the mapping of the
    initial value used to initialize the privatized copies. Shares the
    `acc_firstprivate` user-level clause with `acc.firstprivate`.
    """

    name = "acc.firstprivate_map"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_FIRSTPRIVATE),
        prop_name="dataClause",
    )


@irdl_op_definition
class ReductionOp(_DataEntryOperation):
    """Implementation of upstream acc.reduction.

    The reduction operator (`add`, `mul`, `max`, ...) is carried on the
    `acc.reduction.recipe` referenced via the inherited `recipe`
    SymbolRefAttr property — not on the data-entry op itself.
    """

    name = "acc.reduction"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_REDUCTION),
        prop_name="dataClause",
    )


# ---------------------------------------------------------------------------
# Exit data-clause ops
# ---------------------------------------------------------------------------
#
# Upstream models the exit data-clause family as two TableGen base classes:
# `OpenACC_DataExitOpWithVarPtr` (for `copyout`, `update_host` — both host
# *and* device pointer) and `OpenACC_DataExitOpNoVarPtr` (for `delete`,
# `detach` — device pointer only). They differ in operand layout and
# assembly format, but share the bounds / async / dataClause / structured /
# implicit / modifiers / name surface. The two mixins below mirror that
# split; concrete leaves inherit one and supply only the per-op
# `dataClause` default.


class _DataExitOperationWithVarPtr(IRDLOperation, ABC):
    """Shared shape for `acc.copyout` / `acc.update_host`.

    Mirrors upstream's `OpenACC_DataExitOpWithVarPtr` td class: an `accVar`
    operand (the device pointer, sourced from a matching data-entry op) plus
    a `var` operand (the host pointer to copy/move to) carrying a `varType`
    attr. The bounds / async groups, all shared properties, the assembly
    format, and the keyword-only `__init__` live here.
    """

    acc_var = operand_def()
    var = operand_def()
    bounds = var_operand_def(DataBoundsType)
    async_operands = var_operand_def(IntegerType | IndexType)

    var_type = prop_def(TypeAttribute, prop_name="varType")
    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    structured = opt_prop_def(
        BoolAttr,
        default_value=IntegerAttr.from_bool(True),
        prop_name="structured",
    )
    implicit = opt_prop_def(
        BoolAttr,
        default_value=IntegerAttr.from_bool(False),
        prop_name="implicit",
    )
    modifiers = opt_prop_def(
        DataClauseModifierAttr,
        default_value=DataClauseModifierAttr(frozenset[DataClauseModifier]()),
        prop_name="modifiers",
    )
    var_name = opt_prop_def(StringAttr, prop_name="name")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (AccVar, Var, DeviceTypeOperandsWithKeywordOnly)

    assembly_format = (
        "custom<AccVar>($acc_var, type($acc_var))"
        " (`bounds` `(` $bounds^ `)`)?"
        " (`async` custom<DeviceTypeOperandsWithKeywordOnly>($async_operands,"
        " type($async_operands), $asyncOperandsDeviceType, $asyncOnly)^)?"
        " `to` custom<Var>($var, type($var), $varType)"
        " attr-dict"
    )

    def __init__(
        self,
        *,
        acc_var: SSAValue | Operation,
        var: SSAValue | Operation,
        bounds: Sequence[SSAValue | Operation] = (),
        async_operands: Sequence[SSAValue | Operation] = (),
        var_type: TypeAttribute | None = None,
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        data_clause: DataClauseAttr | DataClause | None = None,
        structured: BoolAttr | bool | None = None,
        implicit: BoolAttr | bool | None = None,
        modifiers: DataClauseModifierAttr | None = None,
        var_name: StringAttr | str | None = None,
    ) -> None:
        var_value = SSAValue.get(var)
        if var_type is None:
            var_type = cast(TypeAttribute, _default_var_type(var_value.type))

        structured_prop: BoolAttr | None = (
            IntegerAttr.from_bool(structured)
            if isinstance(structured, bool)
            else structured
        )
        implicit_prop: BoolAttr | None = (
            IntegerAttr.from_bool(implicit) if isinstance(implicit, bool) else implicit
        )
        data_clause_prop: DataClauseAttr | None = (
            DataClauseAttr(data_clause)
            if isinstance(data_clause, DataClause)
            else data_clause
        )
        var_name_prop: StringAttr | None = (
            StringAttr(var_name) if isinstance(var_name, str) else var_name
        )

        super().__init__(
            operands=[
                [acc_var],
                [var],
                list(bounds),
                list(async_operands),
            ],
            properties={
                "varType": var_type,
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "dataClause": data_clause_prop,
                "structured": structured_prop,
                "implicit": implicit_prop,
                "modifiers": modifiers,
                "name": var_name_prop,
            },
        )


@irdl_op_definition
class CopyoutOp(_DataExitOperationWithVarPtr):
    """Implementation of upstream acc.copyout."""

    name = "acc.copyout"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_COPYOUT),
        prop_name="dataClause",
    )


@irdl_op_definition
class UpdateHostOp(_DataExitOperationWithVarPtr):
    """Implementation of upstream acc.update_host."""

    name = "acc.update_host"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_UPDATE_HOST),
        prop_name="dataClause",
    )


class _DataExitOperationNoVarPtr(IRDLOperation, ABC):
    """Shared shape for `acc.delete` / `acc.detach`.

    Mirrors upstream's `OpenACC_DataExitOpNoVarPtr` td class: just the
    `accVar` operand (device pointer) plus the shared bounds / async /
    dataClause / structured / implicit / modifiers / name surface. No host
    `var` and no `varType` — these ops do not transfer data, so they don't
    need to know the host-side mapping.
    """

    acc_var = operand_def()
    bounds = var_operand_def(DataBoundsType)
    async_operands = var_operand_def(IntegerType | IndexType)

    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    structured = opt_prop_def(
        BoolAttr,
        default_value=IntegerAttr.from_bool(True),
        prop_name="structured",
    )
    implicit = opt_prop_def(
        BoolAttr,
        default_value=IntegerAttr.from_bool(False),
        prop_name="implicit",
    )
    modifiers = opt_prop_def(
        DataClauseModifierAttr,
        default_value=DataClauseModifierAttr(frozenset[DataClauseModifier]()),
        prop_name="modifiers",
    )
    var_name = opt_prop_def(StringAttr, prop_name="name")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (AccVar, DeviceTypeOperandsWithKeywordOnly)

    assembly_format = (
        "custom<AccVar>($acc_var, type($acc_var))"
        " (`bounds` `(` $bounds^ `)`)?"
        " (`async` custom<DeviceTypeOperandsWithKeywordOnly>($async_operands,"
        " type($async_operands), $asyncOperandsDeviceType, $asyncOnly)^)?"
        " attr-dict"
    )

    def __init__(
        self,
        *,
        acc_var: SSAValue | Operation,
        bounds: Sequence[SSAValue | Operation] = (),
        async_operands: Sequence[SSAValue | Operation] = (),
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        data_clause: DataClauseAttr | DataClause | None = None,
        structured: BoolAttr | bool | None = None,
        implicit: BoolAttr | bool | None = None,
        modifiers: DataClauseModifierAttr | None = None,
        var_name: StringAttr | str | None = None,
    ) -> None:
        structured_prop: BoolAttr | None = (
            IntegerAttr.from_bool(structured)
            if isinstance(structured, bool)
            else structured
        )
        implicit_prop: BoolAttr | None = (
            IntegerAttr.from_bool(implicit) if isinstance(implicit, bool) else implicit
        )
        data_clause_prop: DataClauseAttr | None = (
            DataClauseAttr(data_clause)
            if isinstance(data_clause, DataClause)
            else data_clause
        )
        var_name_prop: StringAttr | None = (
            StringAttr(var_name) if isinstance(var_name, str) else var_name
        )

        super().__init__(
            operands=[
                [acc_var],
                list(bounds),
                list(async_operands),
            ],
            properties={
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "dataClause": data_clause_prop,
                "structured": structured_prop,
                "implicit": implicit_prop,
                "modifiers": modifiers,
                "name": var_name_prop,
            },
        )


@irdl_op_definition
class DeleteOp(_DataExitOperationNoVarPtr):
    """Implementation of upstream acc.delete."""

    name = "acc.delete"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_DELETE),
        prop_name="dataClause",
    )


@irdl_op_definition
class DetachOp(_DataExitOperationNoVarPtr):
    """Implementation of upstream acc.detach."""

    name = "acc.detach"
    data_clause = opt_prop_def(
        DataClauseAttr,
        default_value=DataClauseAttr(DataClause.ACC_DETACH),
        prop_name="dataClause",
    )


# ---------------------------------------------------------------------------
# Standalone unstructured data-movement ops: acc.enter_data, acc.exit_data,
# acc.update.
# ---------------------------------------------------------------------------
#
# None of the three carry a region. `enter_data` and `exit_data` share the
# same operand shape: a single optional `asyncOperand` (no per-device-type
# array), a single optional `waitDevnum`, variadic `waitOperands`, and
# `dataClauseOperands`, plus `async` / `wait` UnitAttrs that mirror the
# bare-keyword spellings; `exit_data` additionally carries a `finalize`
# UnitAttr. `acc.update` instead matches the per-device-type async/wait
# shape used by `acc.parallel` (variadic `asyncOperands` /
# `waitOperands` paired with `*DeviceType` / `*Segments` / `hasWaitDevnum` /
# `*Only` arrays) and carries an `ifPresent` UnitAttr.


@irdl_op_definition
class EnterDataOp(IRDLOperation):
    """
    Implementation of upstream acc.enter_data — the OpenACC enter data directive.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accenter_data-accenterdataop).
    """

    name = "acc.enter_data"

    if_cond = opt_operand_def(I1)
    async_operand = opt_operand_def(IntegerType | IndexType)
    wait_devnum = opt_operand_def(IntegerType | IndexType)
    wait_operands = var_operand_def(IntegerType | IndexType)
    data_clause_operands = var_operand_def()

    async_attr = opt_prop_def(UnitAttr, prop_name="async")
    wait_attr = opt_prop_def(UnitAttr, prop_name="wait")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (
        OperandWithKeywordOnly,
        OperandsWithKeywordOnly,
    )

    assembly_format = (
        "(`if` `(` $if_cond^ `)`)?"
        " (`async` custom<OperandWithKeywordOnly>($async_operand,"
        " type($async_operand), $async)^)?"
        " (`wait_devnum` `(` $wait_devnum^ `:` type($wait_devnum) `)`)?"
        " (`wait` custom<OperandsWithKeywordOnly>($wait_operands,"
        " type($wait_operands), $wait)^)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " attr-dict-with-keyword"
    )

    def __init__(
        self,
        *,
        if_cond: SSAValue | Operation | None = None,
        async_operand: SSAValue | Operation | None = None,
        wait_devnum: SSAValue | Operation | None = None,
        wait_operands: Sequence[SSAValue | Operation] = (),
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        async_attr: UnitAttr | bool = False,
        wait_attr: UnitAttr | bool = False,
    ) -> None:
        async_prop: UnitAttr | None = (
            (UnitAttr() if async_attr else None)
            if isinstance(async_attr, bool)
            else async_attr
        )
        wait_prop: UnitAttr | None = (
            (UnitAttr() if wait_attr else None)
            if isinstance(wait_attr, bool)
            else wait_attr
        )
        super().__init__(
            operands=[
                [if_cond] if if_cond is not None else [],
                [async_operand] if async_operand is not None else [],
                [wait_devnum] if wait_devnum is not None else [],
                wait_operands,
                data_clause_operands,
            ],
            properties={
                "async": async_prop,
                "wait": wait_prop,
            },
        )

    def verify_(self) -> None:
        # Mirrors `acc::EnterDataOp::verify` (2.6.6 Data Enter Directive).
        if not self.data_clause_operands:
            raise VerifyException(
                "at least one operand must be present in dataOperands on "
                "the enter data operation"
            )
        if self.async_operand is not None and self.async_attr is not None:
            raise VerifyException("async attribute cannot appear with asyncOperand")
        if self.wait_operands and self.wait_attr is not None:
            raise VerifyException("wait attribute cannot appear with waitOperands")
        if self.wait_devnum is not None and not self.wait_operands:
            raise VerifyException("wait_devnum cannot appear without waitOperands")
        for operand in self.data_clause_operands:
            if not isinstance(operand.owner, (AttachOp, CreateOp, CopyinOp)):
                raise VerifyException("expect data entry operation as defining op")


@irdl_op_definition
class ExitDataOp(IRDLOperation):
    """
    Implementation of upstream acc.exit_data — the OpenACC exit data directive.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accexit_data-accexitdataop).
    """

    name = "acc.exit_data"

    if_cond = opt_operand_def(I1)
    async_operand = opt_operand_def(IntegerType | IndexType)
    wait_devnum = opt_operand_def(IntegerType | IndexType)
    wait_operands = var_operand_def(IntegerType | IndexType)
    data_clause_operands = var_operand_def()

    async_attr = opt_prop_def(UnitAttr, prop_name="async")
    wait_attr = opt_prop_def(UnitAttr, prop_name="wait")
    finalize = opt_prop_def(UnitAttr, prop_name="finalize")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (
        OperandWithKeywordOnly,
        OperandsWithKeywordOnly,
    )

    assembly_format = (
        "(`if` `(` $if_cond^ `)`)?"
        " (`async` custom<OperandWithKeywordOnly>($async_operand,"
        " type($async_operand), $async)^)?"
        " (`wait_devnum` `(` $wait_devnum^ `:` type($wait_devnum) `)`)?"
        " (`wait` custom<OperandsWithKeywordOnly>($wait_operands,"
        " type($wait_operands), $wait)^)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " attr-dict-with-keyword"
    )

    def __init__(
        self,
        *,
        if_cond: SSAValue | Operation | None = None,
        async_operand: SSAValue | Operation | None = None,
        wait_devnum: SSAValue | Operation | None = None,
        wait_operands: Sequence[SSAValue | Operation] = (),
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        async_attr: UnitAttr | bool = False,
        wait_attr: UnitAttr | bool = False,
        finalize: UnitAttr | bool = False,
    ) -> None:
        async_prop: UnitAttr | None = (
            (UnitAttr() if async_attr else None)
            if isinstance(async_attr, bool)
            else async_attr
        )
        wait_prop: UnitAttr | None = (
            (UnitAttr() if wait_attr else None)
            if isinstance(wait_attr, bool)
            else wait_attr
        )
        finalize_prop: UnitAttr | None = (
            (UnitAttr() if finalize else None)
            if isinstance(finalize, bool)
            else finalize
        )
        super().__init__(
            operands=[
                [if_cond] if if_cond is not None else [],
                [async_operand] if async_operand is not None else [],
                [wait_devnum] if wait_devnum is not None else [],
                wait_operands,
                data_clause_operands,
            ],
            properties={
                "async": async_prop,
                "wait": wait_prop,
                "finalize": finalize_prop,
            },
        )

    def verify_(self) -> None:
        # Mirrors `acc::ExitDataOp::verify` (2.6.6 Data Exit Directive).
        # Note that, unlike `EnterDataOp`, upstream does *not* restrict the
        # set of defining ops for `dataClauseOperands` here — the data-exit
        # family is more permissive (copyout / delete / detach / update_host
        # / getdeviceptr all flow in).
        if not self.data_clause_operands:
            raise VerifyException(
                "at least one operand must be present in dataOperands on "
                "the exit data operation"
            )
        if self.async_operand is not None and self.async_attr is not None:
            raise VerifyException("async attribute cannot appear with asyncOperand")
        if self.wait_operands and self.wait_attr is not None:
            raise VerifyException("wait attribute cannot appear with waitOperands")
        if self.wait_devnum is not None and not self.wait_operands:
            raise VerifyException("wait_devnum cannot appear without waitOperands")


@irdl_op_definition
class UpdateOp(IRDLOperation):
    """
    Implementation of upstream acc.update — the OpenACC update executable
    directive.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accupdate-accupdateop).

    Unlike `acc.enter_data` / `acc.exit_data`, `acc.update` carries the same
    per-device-type async/wait operand+attribute shape as `acc.parallel`
    (variadic `asyncOperands`/`waitOperands` paired with `*DeviceType`,
    `*Segments`, `hasWaitDevnum`, and `*Only` arrays). It additionally
    carries an `ifPresent` UnitAttr but no `async` / `wait` keyword-only
    UnitAttrs (those are encoded via the `*Only` device-type arrays).
    """

    name = "acc.update"

    if_cond = opt_operand_def(I1)
    async_operands = var_operand_def(IntegerType | IndexType)
    wait_operands = var_operand_def(IntegerType | IndexType)
    data_clause_operands = var_operand_def()

    async_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="asyncOperandsDeviceType"
    )
    async_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="asyncOnly")
    wait_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="waitOperandsDeviceType"
    )
    wait_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="waitOperandsSegments"
    )
    has_wait_devnum = opt_prop_def(ArrayAttr[BoolAttr], prop_name="hasWaitDevnum")
    wait_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="waitOnly")
    if_present = opt_prop_def(UnitAttr, prop_name="ifPresent")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (
        DeviceTypeOperandsWithKeywordOnly,
        WaitClause,
    )

    assembly_format = (
        "(`if` `(` $if_cond^ `)`)?"
        " (`async` custom<DeviceTypeOperandsWithKeywordOnly>($async_operands,"
        " type($async_operands), $asyncOperandsDeviceType, $asyncOnly)^)?"
        " (`wait` custom<WaitClause>($wait_operands, type($wait_operands),"
        " $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,"
        " $waitOnly)^)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " attr-dict-with-keyword"
    )

    def __init__(
        self,
        *,
        if_cond: SSAValue | Operation | None = None,
        async_operands: Sequence[SSAValue | Operation] = (),
        wait_operands: Sequence[SSAValue | Operation] = (),
        data_clause_operands: Sequence[SSAValue | Operation] = (),
        async_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        async_only: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        wait_operands_segments: DenseArrayBase | None = None,
        has_wait_devnum: ArrayAttr[BoolAttr] | None = None,
        wait_only: ArrayAttr[DeviceTypeAttr] | None = None,
        if_present: UnitAttr | bool = False,
    ) -> None:
        if_present_prop: UnitAttr | None = (
            (UnitAttr() if if_present else None)
            if isinstance(if_present, bool)
            else if_present
        )
        super().__init__(
            operands=[
                [if_cond] if if_cond is not None else [],
                async_operands,
                wait_operands,
                data_clause_operands,
            ],
            properties={
                "asyncOperandsDeviceType": async_operands_device_type,
                "asyncOnly": async_only,
                "waitOperandsDeviceType": wait_operands_device_type,
                "waitOperandsSegments": wait_operands_segments,
                "hasWaitDevnum": has_wait_devnum,
                "waitOnly": wait_only,
                "ifPresent": if_present_prop,
            },
        )

    def verify_(self) -> None:
        # Mirrors `acc::UpdateOp::verify`.
        if not self.data_clause_operands:
            raise VerifyException("at least one value must be present in dataOperands")
        for operand in self.data_clause_operands:
            if not isinstance(
                operand.owner, (UpdateDeviceOp, UpdateHostOp, GetDevicePtrOp)
            ):
                raise VerifyException(
                    "expect data entry/exit operation or acc.getdeviceptr "
                    "as defining op"
                )


# ---------------------------------------------------------------------------
# Atomic family
# ---------------------------------------------------------------------------
#
# `acc.atomic.read` and `acc.atomic.write` are the leaf atomic ops. Each has
# two pointer-like operands plus an optional `if(%cond)` clause shared with
# the rest of the family via the `AtomicIfClause` custom directive.
# `acc.atomic.update` and `acc.atomic.capture` follow in subsequent PRs.


@irdl_op_definition
class AtomicReadOp(IRDLOperation):
    """
    Implementation of upstream acc.atomic.read — performs an atomic read.

    The operand `x` is the address from where the value is atomically read.
    The operand `v` is the address where the value is stored after reading.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accatomicread-accatomicreadop).
    """

    name = "acc.atomic.read"

    x = operand_def()
    v = operand_def()
    if_cond = opt_operand_def(I1)

    element_type = prop_def(TypeAttribute)

    irdl_options = (ParsePropInAttrDict(),)

    custom_directives = (AtomicIfClause,)

    assembly_format = (
        "custom<AtomicIfClause>($if_cond, type($if_cond))"
        " $v `=` $x `:` type($v) `,` type($x) `,` $element_type attr-dict"
    )

    def __init__(
        self,
        *,
        x: SSAValue | Operation,
        v: SSAValue | Operation,
        element_type: TypeAttribute,
        if_cond: SSAValue | Operation | None = None,
    ) -> None:
        super().__init__(
            operands=[x, v, if_cond],
            properties={"element_type": element_type},
        )

    def verify_(self) -> None:
        # Mirrors upstream `AtomicReadOpInterface::verifyCommon`.
        if self.x is self.v:
            raise VerifyException(
                "read and write must not be to the same location for atomic reads"
            )


@irdl_op_definition
class AtomicWriteOp(IRDLOperation):
    """
    Implementation of upstream acc.atomic.write — performs an atomic write.

    The operand `x` is the address to where `expr` is atomically written.
    In general the type of `x` must dereference to the type of `expr`.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accatomicwrite-accatomicwriteop).
    """

    name = "acc.atomic.write"

    x = operand_def()
    expr = operand_def()
    if_cond = opt_operand_def(I1)

    custom_directives = (AtomicIfClause,)

    assembly_format = (
        "custom<AtomicIfClause>($if_cond, type($if_cond))"
        " $x `=` $expr `:` type($x) `,` type($expr) attr-dict"
    )

    def __init__(
        self,
        *,
        x: SSAValue | Operation,
        expr: SSAValue | Operation,
        if_cond: SSAValue | Operation | None = None,
    ) -> None:
        super().__init__(operands=[x, expr, if_cond])

    def verify_(self) -> None:
        # Mirrors upstream `AtomicWriteOpInterface::verifyCommon`: the
        # pointee type of `x` (currently restricted to `MemRefType`) must
        # match `expr`'s type. Non-memref pointer-likes are not yet
        # modelled in xDSL's `acc` dialect; when they land, extend the
        # element-type lookup accordingly.
        x_type = self.x.type
        if isa(x_type, MemRefType):
            if x_type.element_type != self.expr.type:
                raise VerifyException("address must dereference to value type")


@irdl_op_definition
class AtomicUpdateOp(IRDLOperation):
    """
    Implementation of upstream acc.atomic.update — performs an atomic update.

    The operand `x` is the address of the variable being updated. The region
    describes how to update the value: it takes the current value at `x` as
    its single block argument and must yield the updated value via acc.yield.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accatomicupdate-accatomicupdateop).
    """

    name = "acc.atomic.update"

    x = operand_def()
    if_cond = opt_operand_def(I1)

    region = region_def("single_block")

    custom_directives = (AtomicIfClause,)

    assembly_format = (
        "custom<AtomicIfClause>($if_cond, type($if_cond))"
        " $x `:` type($x) $region attr-dict"
    )

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(YieldOp),
            RecursiveMemoryEffect(),
        )
    )

    def __init__(
        self,
        *,
        x: SSAValue | Operation,
        region: Region,
        if_cond: SSAValue | Operation | None = None,
    ) -> None:
        super().__init__(
            operands=[x, if_cond],
            regions=[region],
        )

    def verify_(self) -> None:
        # Mirrors upstream `AtomicUpdateOpInterface::verifyCommon` +
        # `verifyRegionsCommon`. The terminator-kind / single-block /
        # non-empty-block checks are handled by the
        # `SingleBlockImplicitTerminator(YieldOp)` trait.
        block = self.region.block
        if len(block.args) != 1:
            raise VerifyException("the region must accept exactly one argument")

        arg_type = block.args[0].type
        x_type = self.x.type
        if isa(x_type, MemRefType) and x_type.element_type != arg_type:
            raise VerifyException(
                "the type of the operand must be a pointer type whose "
                "element type is the same as that of the region argument"
            )

        terminator = cast(YieldOp, block.last_op)
        if len(terminator.operands) != 1:
            raise VerifyException("only updated value must be returned")
        if terminator.operands[0].type != arg_type:
            raise VerifyException("input and yielded value must have the same type")


@irdl_op_definition
class AtomicCaptureOp(IRDLOperation):
    """
    Implementation of upstream acc.atomic.capture — performs an atomic capture.

    The region contains exactly two atomic ops (plus an implicit acc.terminator)
    in one of three allowed orderings: `update + read`, `read + update`, or
    `read + write`. The two operations must operate on the same memory
    location.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accatomiccapture-accatomiccaptureop).
    """

    name = "acc.atomic.capture"

    if_cond = opt_operand_def(I1)

    region = region_def("single_block")

    traits = lazy_traits_def(
        lambda: (
            SingleBlockImplicitTerminator(TerminatorOp),
            RecursiveMemoryEffect(),
        )
    )

    @classmethod
    def parse(cls, parser: Parser) -> "AtomicCaptureOp":
        # Mirror upstream `oilist( \`if\` \`(\` $ifCond \`)\` ) $region attr-dict`
        # plus `SingleBlockImplicitTerminator(TerminatorOp)`: accept the body
        # without an explicit `acc.terminator` and insert one.
        if_cond: SSAValue | None = None
        if parser.parse_optional_keyword("if") is not None:
            parser.parse_punctuation("(")
            unresolved = parser.parse_unresolved_operand()
            parser.parse_punctuation(")")
            if_cond = parser.resolve_operand(unresolved, i1)
        region = parser.parse_region()
        attrs = parser.parse_optional_attr_dict()
        op = cls(region=region, if_cond=if_cond)
        op.attributes = {**op.attributes, **attrs}
        for trait in op.get_traits_of_type(SingleBlockImplicitTerminator):
            ensure_terminator(op, trait)
        return op

    def print(self, printer: Printer) -> None:
        if self.if_cond is not None:
            printer.print_string(" if(")
            printer.print_ssa_value(self.if_cond)
            printer.print_string(")")
        printer.print_string(" ")
        # Strip the implicit `acc.terminator` to mirror upstream's print
        # behavior. The `SingleBlockImplicitTerminator` trait re-inserts
        # one on parse, so this round-trips through both xdsl-opt and
        # mlir-opt bit-identically.
        printer.print_region(self.region, print_block_terminators=False)
        printer.print_op_attributes(self.attributes)

    def __init__(
        self,
        *,
        region: Region,
        if_cond: SSAValue | Operation | None = None,
    ) -> None:
        super().__init__(
            operands=[if_cond],
            regions=[region],
        )

    def verify_(self) -> None:
        # Mirrors upstream AtomicCaptureOpInterface::verifyRegionsCommon.
        # SingleBlockImplicitTerminator(TerminatorOp) guarantees the last
        # op is an acc.terminator, so the valid op count is exactly 3:
        # two atomic ops plus the terminator.
        ops = list(self.region.block.ops)
        if len(ops) != 3:
            raise VerifyException(
                "expected three operations in atomic.capture region (one "
                "terminator, and two atomic ops)"
            )
        first_op, second_op, _terminator = ops
        if isinstance(first_op, AtomicUpdateOp) and isinstance(second_op, AtomicReadOp):
            if first_op.x is not second_op.x:
                raise VerifyException(
                    "updated variable in atomic.update must be captured in "
                    "second operation"
                )
        elif isinstance(first_op, AtomicReadOp) and isinstance(
            second_op, AtomicUpdateOp
        ):
            if first_op.x is not second_op.x:
                raise VerifyException(
                    "captured variable in atomic.read must be updated in "
                    "second operation"
                )
        elif isinstance(first_op, AtomicReadOp) and isinstance(
            second_op, AtomicWriteOp
        ):
            if first_op.x is not second_op.x:
                raise VerifyException(
                    "captured variable in atomic.read must be updated in "
                    "second operation"
                )
        else:
            raise VerifyException(
                "invalid sequence of operations in the capture region"
            )


# ---------------------------------------------------------------------------
# Declare family
# ---------------------------------------------------------------------------
#
# `acc.declare_enter`, `acc.declare_exit`, and `acc.declare` all carry a
# variadic `dataClauseOperands` and validate it the same way: every operand
# must be defined by one of the eight data ops listed in upstream's
# `checkDeclareOperands` helper. The diagnostic strings ("at least one
# operand must appear on the declare operation" / "expect valid declare data
# entry operation or acc.getdeviceptr as defining op") are copied verbatim
# from `mlir/lib/Dialect/OpenACC/IR/OpenACC.cpp` so external tooling and
# upstream tests round-trip identically.


def _verify_declare_operands(
    operands: Sequence[SSAValue], *, require_at_least_one: bool = True
) -> None:
    """Mirror of upstream's `checkDeclareOperands` template helper.

    `acc.declare_exit` passes ``require_at_least_one=False`` when a `token`
    is present (the token already pins the implicit data region — operands
    are then redundant); every other caller requires at least one operand.
    """
    if not operands and require_at_least_one:
        raise VerifyException(
            "at least one operand must appear on the declare operation"
        )
    for operand in operands:
        if not isinstance(
            operand.owner,
            (
                CopyinOp,
                CopyoutOp,
                CreateOp,
                DevicePtrOp,
                GetDevicePtrOp,
                PresentOp,
                DeclareDeviceResidentOp,
                DeclareLinkOp,
            ),
        ):
            raise VerifyException(
                "expect valid declare data entry operation or "
                "acc.getdeviceptr as defining op"
            )


@irdl_op_definition
class DeclareEnterOp(IRDLOperation):
    """
    Implementation of upstream acc.declare_enter — entry to an implicit
    OpenACC declare data region. Yields an `!acc.declare_token` that can
    optionally be threaded into the matching `acc.declare_exit`.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accdeclare_enter-accdeclareenterop).
    """

    name = "acc.declare_enter"

    data_clause_operands = var_operand_def()

    token = result_def(DeclareTokenType)

    assembly_format = (
        "(`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " attr-dict-with-keyword"
    )

    def __init__(
        self,
        *,
        data_clause_operands: Sequence[SSAValue | Operation] = (),
    ) -> None:
        super().__init__(
            operands=[data_clause_operands],
            result_types=[DeclareTokenType()],
        )

    def verify_(self) -> None:
        _verify_declare_operands(self.data_clause_operands)


@irdl_op_definition
class DeclareExitOp(IRDLOperation):
    """
    Implementation of upstream acc.declare_exit — exit from an implicit
    OpenACC declare data region, optionally consuming the
    `!acc.declare_token` produced by the matching `acc.declare_enter`.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accdeclare_exit-accdeclareexitop).
    """

    name = "acc.declare_exit"

    token = opt_operand_def(DeclareTokenType)
    data_clause_operands = var_operand_def()

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    assembly_format = (
        "(`token` `(` $token^ `)`)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " attr-dict-with-keyword"
    )

    def __init__(
        self,
        *,
        token: SSAValue | Operation | None = None,
        data_clause_operands: Sequence[SSAValue | Operation] = (),
    ) -> None:
        super().__init__(
            operands=[
                [token] if token is not None else [],
                data_clause_operands,
            ],
        )

    def verify_(self) -> None:
        # Upstream `acc::DeclareExitOp::verify`: the `token`-bearing form
        # relaxes the at-least-one operand requirement.
        _verify_declare_operands(
            self.data_clause_operands,
            require_at_least_one=self.token is None,
        )


@irdl_op_definition
class DeclareOp(IRDLOperation):
    """
    Implementation of upstream acc.declare — the structured declare region
    inside a function/subroutine. Body is an arbitrary single-block region
    (no terminator) holding the implicit data region's lifetime.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accdeclare-accdeclareop).
    """

    name = "acc.declare"

    data_clause_operands = var_operand_def()

    region = region_def()

    assembly_format = (
        "(`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " $region attr-dict-with-keyword"
    )

    traits = traits_def(NoTerminator(), RecursiveMemoryEffect())

    def __init__(
        self,
        *,
        region: Region,
        data_clause_operands: Sequence[SSAValue | Operation] = (),
    ) -> None:
        super().__init__(
            operands=[data_clause_operands],
            regions=[region],
        )

    def verify_(self) -> None:
        _verify_declare_operands(self.data_clause_operands)


@irdl_op_definition
class DataBoundsOp(IRDLOperation):
    """
    Implementation of upstream acc.bounds.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accbounds-accdataboundsop).

    The verifier requires either `extent` or `upperbound` (or both) to be
    present. The five operand groups are gated by `AttrSizedOperandSegments`,
    matching upstream's `Optional<IntOrIndex>` argument shape.
    """

    name = "acc.bounds"

    lowerbound = opt_operand_def(IntegerType | IndexType)
    upperbound = opt_operand_def(IntegerType | IndexType)
    extent = opt_operand_def(IntegerType | IndexType)
    stride = opt_operand_def(IntegerType | IndexType)
    start_idx = opt_operand_def(IntegerType | IndexType)

    stride_in_bytes = opt_prop_def(BoolAttr, prop_name="strideInBytes")

    result = result_def(DataBoundsType)

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    # Upstream uses `oilist(...)` so each clause is independently optional and
    # may appear in any order. xDSL's declarative format requires a fixed
    # sequence; we follow the upstream td definition order
    # (lowerbound, upperbound, extent, stride, startIdx). This still matches
    # what mlir-opt prints, since oilist also emits in td order.
    assembly_format = (
        "(`lowerbound` `(` $lowerbound^ `:` type($lowerbound) `)`)?"
        " (`upperbound` `(` $upperbound^ `:` type($upperbound) `)`)?"
        " (`extent` `(` $extent^ `:` type($extent) `)`)?"
        " (`stride` `(` $stride^ `:` type($stride) `)`)?"
        " (`startIdx` `(` $start_idx^ `:` type($start_idx) `)`)?"
        " attr-dict"
    )

    traits = lazy_traits_def(lambda: (NoMemoryEffect(),))

    def __init__(
        self,
        *,
        lowerbound: SSAValue | Operation | None = None,
        upperbound: SSAValue | Operation | None = None,
        extent: SSAValue | Operation | None = None,
        stride: SSAValue | Operation | None = None,
        start_idx: SSAValue | Operation | None = None,
        stride_in_bytes: BoolAttr | None = None,
    ) -> None:
        super().__init__(
            operands=[
                [lowerbound] if lowerbound is not None else [],
                [upperbound] if upperbound is not None else [],
                [extent] if extent is not None else [],
                [stride] if stride is not None else [],
                [start_idx] if start_idx is not None else [],
            ],
            properties={"strideInBytes": stride_in_bytes},
            result_types=[DataBoundsType()],
        )

    def verify_(self) -> None:
        if self.extent is None and self.upperbound is None:
            raise VerifyException("expected extent or upperbound.")


class _DataBoundsAccessorOp(IRDLOperation, ABC):
    """
    Shared shape for the `acc.get_lowerbound` / `get_upperbound` /
    `get_stride` / `get_extent` accessor ops. Each takes an
    `!acc.data_bounds_ty` operand and yields an `index`.
    """

    bounds = operand_def(DataBoundsType)
    result = result_def(IndexType)

    assembly_format = "$bounds attr-dict `:` `(` type($bounds) `)` `->` type($result)"

    traits = lazy_traits_def(lambda: (NoMemoryEffect(),))

    def __init__(self, bounds: SSAValue | Operation) -> None:
        super().__init__(operands=[bounds], result_types=[IndexType()])


@irdl_op_definition
class GetLowerboundOp(_DataBoundsAccessorOp):
    """Implementation of upstream acc.get_lowerbound."""

    name = "acc.get_lowerbound"


@irdl_op_definition
class GetUpperboundOp(_DataBoundsAccessorOp):
    """Implementation of upstream acc.get_upperbound."""

    name = "acc.get_upperbound"


@irdl_op_definition
class GetStrideOp(_DataBoundsAccessorOp):
    """Implementation of upstream acc.get_stride."""

    name = "acc.get_stride"


@irdl_op_definition
class GetExtentOp(_DataBoundsAccessorOp):
    """Implementation of upstream acc.get_extent."""

    name = "acc.get_extent"


# ---------------------------------------------------------------------------
# Privatization / reduction recipes
# ---------------------------------------------------------------------------
#
# `acc.private.recipe`, `acc.firstprivate.recipe`, and
# `acc.reduction.recipe` declare reusable templates for how to allocate /
# initialize / copy / combine / destroy a privatized or reduction value.
# Each is a top-level symbol op (`Symbol` + `IsolatedFromAbove`); the
# corresponding data-clause ops reference the recipe by name.
#
# The three ops share `sym_name` + `type` props, so we factor that into
# the abstract `_RecipeOperation` mixin; concrete leaves declare only their
# own regions, traits, assembly format, and per-op `verify_`.
# `acc.reduction.recipe` additionally carries a `reductionOperator`
# property typed against `ReductionOpKindAttr`.


def _verify_init_like_region(
    region: Region,
    region_type: str,
    region_name: str,
    expected_type: Attribute,
    *,
    optional: bool = False,
) -> None:
    """Port of upstream `verifyInitLikeSingleArgRegion`.

    The init / destroy region must be non-empty (unless `optional`) and the
    first argument of its first block must be of the recipe's type.
    """
    if optional and not region.blocks:
        return
    if not region.blocks:
        raise VerifyException(f"expects non-empty {region_name} region")
    first_block = region.block
    if not first_block.args or first_block.args[0].type != expected_type:
        raise VerifyException(
            f"expects {region_name} region first argument of the {region_type} type"
        )


class _RecipeOperation(IRDLOperation, ABC):
    """Shared `sym_name` + `type` shape for the recipe ops.

    Mirrors upstream's near-identical `OpenACC_*RecipeOp` td classes.
    Concrete leaves declare their own regions, traits, assembly format, and
    per-op `verify_`.
    """

    sym_name = prop_def(SymbolNameConstraint())
    var_type = prop_def(TypeAttribute, prop_name="type")


@irdl_op_definition
class PrivateRecipeOp(_RecipeOperation):
    """
    Implementation of upstream acc.private.recipe.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accprivaterecipe-accprivaterecipeop).
    """

    name = "acc.private.recipe"

    init_region = region_def()
    destroy_region = region_def()

    traits = lazy_traits_def(lambda: (IsolatedFromAbove(), SymbolOpInterface()))

    assembly_format = (
        "$sym_name `:` $type attr-dict-with-keyword"
        " `init` $init_region"
        " (`destroy` $destroy_region^)?"
    )

    def __init__(
        self,
        *,
        sym_name: StringAttr | str,
        var_type: TypeAttribute,
        init_region: Region,
        destroy_region: Region | None = None,
    ) -> None:
        sym_name_attr: StringAttr = (
            StringAttr(sym_name) if isinstance(sym_name, str) else sym_name
        )
        super().__init__(
            properties={
                "sym_name": sym_name_attr,
                "type": var_type,
            },
            regions=[init_region, destroy_region or Region()],
        )

    def verify_(self) -> None:
        _verify_init_like_region(
            self.init_region,
            "privatization",
            "init",
            self.var_type,
        )
        _verify_init_like_region(
            self.destroy_region,
            "privatization",
            "destroy",
            self.var_type,
            optional=True,
        )


@irdl_op_definition
class FirstprivateRecipeOp(_RecipeOperation):
    """
    Implementation of upstream acc.firstprivate.recipe.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accfirstprivaterecipe-accfirstprivaterecipeop).
    """

    name = "acc.firstprivate.recipe"

    init_region = region_def()
    copy_region = region_def()
    destroy_region = region_def()

    traits = lazy_traits_def(lambda: (IsolatedFromAbove(), SymbolOpInterface()))

    assembly_format = (
        "$sym_name `:` $type attr-dict-with-keyword"
        " `init` $init_region"
        " `copy` $copy_region"
        " (`destroy` $destroy_region^)?"
    )

    def __init__(
        self,
        *,
        sym_name: StringAttr | str,
        var_type: TypeAttribute,
        init_region: Region,
        copy_region: Region,
        destroy_region: Region | None = None,
    ) -> None:
        sym_name_attr: StringAttr = (
            StringAttr(sym_name) if isinstance(sym_name, str) else sym_name
        )
        super().__init__(
            properties={
                "sym_name": sym_name_attr,
                "type": var_type,
            },
            regions=[init_region, copy_region, destroy_region or Region()],
        )

    def verify_(self) -> None:
        _verify_init_like_region(
            self.init_region,
            "privatization",
            "init",
            self.var_type,
        )
        if not self.copy_region.blocks:
            raise VerifyException("expects non-empty copy region")
        copy_block = self.copy_region.block
        if (
            len(copy_block.args) < 2
            or copy_block.args[0].type != self.var_type
            or copy_block.args[1].type != self.var_type
        ):
            raise VerifyException(
                "expects copy region with two arguments of the privatization type"
            )
        if self.destroy_region.blocks:
            _verify_init_like_region(
                self.destroy_region,
                "privatization",
                "destroy",
                self.var_type,
            )


@irdl_op_definition
class ReductionRecipeOp(_RecipeOperation):
    """
    Implementation of upstream acc.reduction.recipe.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accreductionrecipe-accreductionrecipeop).
    """

    name = "acc.reduction.recipe"

    reduction_operator = prop_def(ReductionOpKindAttr, prop_name="reductionOperator")

    init_region = region_def()
    combiner_region = region_def()
    destroy_region = region_def()

    traits = lazy_traits_def(lambda: (IsolatedFromAbove(), SymbolOpInterface()))

    # Note: `$reductionOperator` prints/parses as `<value>` (just the
    # parameter, no dialect prefix) because `ReductionOpKindAttr` overrides
    # `print_parameter`/`parse_parameter` to wrap the value in angle
    # brackets — matching upstream's `assemblyFormat = "`<` $value `>`"`.
    # The declarative format machinery uses `UniqueBaseAttributeVariable`
    # for non-builtin attrs, which calls `print_parameter` directly and
    # bypasses the `#acc.reduction_operator` opaque prefix.
    assembly_format = (
        "$sym_name `:` $type attr-dict-with-keyword"
        " `reduction_operator` $reductionOperator"
        " `init` $init_region"
        " `combiner` $combiner_region"
        " (`destroy` $destroy_region^)?"
    )

    def __init__(
        self,
        *,
        sym_name: StringAttr | str,
        var_type: TypeAttribute,
        reduction_operator: ReductionOpKindAttr | ReductionOpKind,
        init_region: Region,
        combiner_region: Region,
        destroy_region: Region | None = None,
    ) -> None:
        sym_name_attr: StringAttr = (
            StringAttr(sym_name) if isinstance(sym_name, str) else sym_name
        )
        operator_attr: ReductionOpKindAttr = (
            ReductionOpKindAttr(reduction_operator)
            if isinstance(reduction_operator, ReductionOpKind)
            else reduction_operator
        )
        super().__init__(
            properties={
                "sym_name": sym_name_attr,
                "type": var_type,
                "reductionOperator": operator_attr,
            },
            regions=[init_region, combiner_region, destroy_region or Region()],
        )

    def verify_(self) -> None:
        _verify_init_like_region(
            self.init_region,
            "reduction",
            "init",
            self.var_type,
        )
        if not self.combiner_region.blocks:
            raise VerifyException("expects non-empty combiner region")
        combiner_block = self.combiner_region.block
        if (
            len(combiner_block.args) < 2
            or combiner_block.args[0].type != self.var_type
            or combiner_block.args[1].type != self.var_type
        ):
            raise VerifyException(
                "expects combiner region with the first two arguments of the "
                "reduction type"
            )
        for yield_op in self.combiner_region.walk():
            if not isinstance(yield_op, YieldOp):
                continue
            if (
                len(yield_op.arguments) != 1
                or yield_op.arguments[0].type != self.var_type
            ):
                raise VerifyException(
                    "expects combiner region to yield a value of the reduction type"
                )
        if self.destroy_region.blocks:
            _verify_init_like_region(
                self.destroy_region,
                "reduction",
                "destroy",
                self.var_type,
            )


# ---------------------------------------------------------------------------
# Runtime executable ops: acc.init, acc.shutdown, acc.set, acc.wait
# ---------------------------------------------------------------------------
#
# Upstream models acc.init / acc.shutdown with identical surface — same two
# optional operands (device_num, if_cond), same `device_types` ArrayAttr
# property, same assembly format, and the same "cannot be nested in a
# compute operation" verifier (`mlir/lib/Dialect/OpenACC/IR/OpenACC.cpp`,
# `isComputeOperation`; the set of compute ops is parallel / serial /
# kernels / loop). The xDSL port factors that shape into the
# `_RuntimeDeviceTypesOperation` mixin below; concrete leaves override only `name`.
#
# `acc.set` and `acc.wait` diverge from that shape (different operand
# counts, different property names/types, different assembly formats) so
# they're written as standalone IRDL ops. Both still share the
# `_verify_not_in_compute_op` parent walk — except `acc.wait`, which
# upstream explicitly leaves unrestricted (see `acc::WaitOp::verify`).


def _verify_not_in_compute_op(op: IRDLOperation) -> None:
    """Mirrors upstream's `isComputeOperation` parent walk."""
    parent = op.parent_op()
    while parent is not None:
        if isinstance(parent, (ParallelOp, SerialOp, KernelsOp, LoopOp)):
            raise VerifyException(
                f"'{op.name}' op cannot be nested in a compute operation"
            )
        parent = parent.parent_op()


class _RuntimeDeviceTypesOperation(IRDLOperation, ABC):
    """Base class for `acc.init` / `acc.shutdown`.

    Concrete leaves inherit every IRDL field, the assembly format, the
    `__init__`, and `verify_` — they only override `name`.
    """

    device_num = opt_operand_def(IntegerType | IndexType)
    if_cond = opt_operand_def(I1)

    device_types = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="device_types")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    assembly_format = (
        "(`device_num` `(` $device_num^ `:` type($device_num) `)`)?"
        " (`if` `(` $if_cond^ `)`)?"
        " attr-dict-with-keyword"
    )

    def __init__(
        self,
        *,
        device_num: SSAValue | Operation | None = None,
        if_cond: SSAValue | Operation | None = None,
        device_types: ArrayAttr[DeviceTypeAttr] | None = None,
    ) -> None:
        super().__init__(
            operands=[device_num, if_cond],
            properties={"device_types": device_types},
        )

    def verify_(self) -> None:
        _verify_not_in_compute_op(self)


@irdl_op_definition
class InitOp(_RuntimeDeviceTypesOperation):
    """
    Implementation of upstream acc.init — the OpenACC init executable directive.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accinit-accinitop).
    """

    name = "acc.init"


@irdl_op_definition
class ShutdownOp(_RuntimeDeviceTypesOperation):
    """
    Implementation of upstream acc.shutdown — the OpenACC shutdown executable
    directive.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accshutdown-accshutdownop).
    """

    name = "acc.shutdown"


@irdl_op_definition
class SetOp(IRDLOperation):
    """
    Implementation of upstream acc.set — the OpenACC set executable directive.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accset-accsetop).
    """

    name = "acc.set"

    default_async = opt_operand_def(IntegerType | IndexType)
    device_num = opt_operand_def(IntegerType | IndexType)
    if_cond = opt_operand_def(I1)

    device_type = opt_prop_def(DeviceTypeAttr, prop_name="device_type")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    assembly_format = (
        "(`default_async` `(` $default_async^ `:` type($default_async) `)`)?"
        " (`device_num` `(` $device_num^ `:` type($device_num) `)`)?"
        " (`if` `(` $if_cond^ `)`)?"
        " attr-dict-with-keyword"
    )

    def __init__(
        self,
        *,
        default_async: SSAValue | Operation | None = None,
        device_num: SSAValue | Operation | None = None,
        if_cond: SSAValue | Operation | None = None,
        device_type: DeviceTypeAttr | None = None,
    ) -> None:
        super().__init__(
            operands=[default_async, device_num, if_cond],
            properties={"device_type": device_type},
        )

    def verify_(self) -> None:
        _verify_not_in_compute_op(self)
        if (
            self.device_type is None
            and self.default_async is None
            and self.device_num is None
        ):
            raise VerifyException(
                "at least one default_async, device_num, or device_type "
                "operand must appear"
            )


@irdl_op_definition
class WaitOp(IRDLOperation):
    """
    Implementation of upstream acc.wait — the OpenACC wait executable directive.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accwait-accwaitop).
    """

    name = "acc.wait"

    wait_operands = var_operand_def(IntegerType | IndexType)
    async_operand = opt_operand_def(IntegerType | IndexType)
    wait_devnum = opt_operand_def(IntegerType | IndexType)
    if_cond = opt_operand_def(I1)

    async_attr = opt_prop_def(UnitAttr, prop_name="async")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        ParsePropInAttrDict(),
    )

    custom_directives = (OperandWithKeywordOnly,)

    assembly_format = (
        "( `(` $wait_operands^ `:` type($wait_operands) `)` )?"
        " (`async` custom<OperandWithKeywordOnly>($async_operand,"
        " type($async_operand), $async)^)?"
        " (`wait_devnum` `(` $wait_devnum^ `:` type($wait_devnum) `)`)?"
        " (`if` `(` $if_cond^ `)`)?"
        " attr-dict-with-keyword"
    )

    def __init__(
        self,
        *,
        wait_operands: Sequence[SSAValue | Operation] = (),
        async_operand: SSAValue | Operation | None = None,
        wait_devnum: SSAValue | Operation | None = None,
        if_cond: SSAValue | Operation | None = None,
        async_attr: UnitAttr | None = None,
    ) -> None:
        super().__init__(
            operands=[wait_operands, async_operand, wait_devnum, if_cond],
            properties={"async": async_attr},
        )

    def verify_(self) -> None:
        # Mirrors upstream `acc::WaitOp::verify`. Note that — unlike
        # init/shutdown/set — `acc.wait` does *not* forbid nesting inside a
        # compute construct.
        if self.async_operand is not None and self.async_attr is not None:
            raise VerifyException("async attribute cannot appear with asyncOperand")
        if self.wait_devnum is not None and not self.wait_operands:
            raise VerifyException("wait_devnum cannot appear without waitOperands")


@irdl_op_definition
class RoutineOp(IRDLOperation):
    """
    Implementation of upstream acc.routine — captures the clauses of an
    OpenACC routine directive together with the associated function name.
    The function keeps track of its corresponding routine declaration via
    the `acc.routine_info` discardable attribute (an `ArrayAttr` of
    `SymbolRefAttr` pointing back at the matching `acc.routine` ops).

    `bind`, `gang`, `worker`, `vector`, and `seq` clauses are tracked
    per-device-type via parallel `ArrayAttr` properties — for any non-`none`
    device type, at most one of `gang`/`worker`/`vector`/`seq` may be set,
    and similarly at most one for the `none` device type.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accroutine-accroutineop).
    """

    name = "acc.routine"

    sym_name = prop_def(SymbolNameConstraint())
    func_name = prop_def(SymbolRefAttr)
    bind_id_name = opt_prop_def(ArrayAttr[SymbolRefAttr], prop_name="bindIdName")
    bind_str_name = opt_prop_def(ArrayAttr[StringAttr], prop_name="bindStrName")
    bind_id_name_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="bindIdNameDeviceType"
    )
    bind_str_name_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="bindStrNameDeviceType"
    )
    worker = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    vector = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    seq = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    nohost = opt_prop_def(UnitAttr)
    implicit = opt_prop_def(UnitAttr)
    gang = opt_prop_def(ArrayAttr[DeviceTypeAttr])
    gang_dim = opt_prop_def(ArrayAttr[IntegerAttr], prop_name="gangDim")
    gang_dim_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="gangDimDeviceType"
    )

    custom_directives = (BindName, RoutineGangClause, DeviceTypeArrayClause)

    assembly_format = (
        "$sym_name `func` `(` $func_name `)`"
        " (`bind` `(` custom<BindName>($bindIdName, $bindStrName,"
        " $bindIdNameDeviceType, $bindStrNameDeviceType)^ `)`)?"
        " (`gang` custom<RoutineGangClause>($gang, $gangDim,"
        " $gangDimDeviceType)^)?"
        " (`worker` custom<DeviceTypeArrayClause>($worker)^)?"
        " (`vector` custom<DeviceTypeArrayClause>($vector)^)?"
        " (`seq` custom<DeviceTypeArrayClause>($seq)^)?"
        " (`nohost` $nohost^)?"
        " (`implicit` $implicit^)?"
        " attr-dict-with-keyword"
    )

    traits = lazy_traits_def(lambda: (IsolatedFromAbove(), SymbolOpInterface()))

    def __init__(
        self,
        *,
        sym_name: StringAttr | str,
        func_name: SymbolRefAttr | str,
        bind_id_name: ArrayAttr[SymbolRefAttr] | None = None,
        bind_str_name: ArrayAttr[StringAttr] | None = None,
        bind_id_name_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        bind_str_name_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
        worker: ArrayAttr[DeviceTypeAttr] | None = None,
        vector: ArrayAttr[DeviceTypeAttr] | None = None,
        seq: ArrayAttr[DeviceTypeAttr] | None = None,
        nohost: UnitAttr | bool = False,
        implicit: UnitAttr | bool = False,
        gang: ArrayAttr[DeviceTypeAttr] | None = None,
        gang_dim: ArrayAttr[IntegerAttr] | None = None,
        gang_dim_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
    ) -> None:
        sym_name_attr: StringAttr = (
            StringAttr(sym_name) if isinstance(sym_name, str) else sym_name
        )
        func_name_attr: SymbolRefAttr = (
            SymbolRefAttr(func_name) if isinstance(func_name, str) else func_name
        )
        nohost_prop: UnitAttr | None = (
            (UnitAttr() if nohost else None) if isinstance(nohost, bool) else nohost
        )
        implicit_prop: UnitAttr | None = (
            (UnitAttr() if implicit else None)
            if isinstance(implicit, bool)
            else implicit
        )
        super().__init__(
            properties={
                "sym_name": sym_name_attr,
                "func_name": func_name_attr,
                "bindIdName": bind_id_name,
                "bindStrName": bind_str_name,
                "bindIdNameDeviceType": bind_id_name_device_type,
                "bindStrNameDeviceType": bind_str_name_device_type,
                "worker": worker,
                "vector": vector,
                "seq": seq,
                "nohost": nohost_prop,
                "implicit": implicit_prop,
                "gang": gang,
                "gangDim": gang_dim,
                "gangDimDeviceType": gang_dim_device_type,
            },
        )

    def verify_(self) -> None:
        # Mirrors `acc::RoutineOp::verify`: at most one of
        # gang/worker/vector/seq may be set per device_type, and a
        # `none`-device entry conflicts with any per-device entry. Gang's
        # presence is recorded in either `gang` (kw-only DT) or in
        # `gangDimDeviceType` (the parallel array for sized gangs).
        bucket_dts: list[set[DeviceType]] = [
            {entry.data for arr in bucket if arr is not None for entry in arr.data}
            for bucket in (
                (self.gang, self.gang_dim_device_type),
                (self.worker,),
                (self.vector,),
                (self.seq,),
            )
        ]

        base = sum(DeviceType.NONE in dts for dts in bucket_dts)
        if base > 1:
            raise VerifyException(
                "only one of `gang`, `worker`, `vector`, `seq` can be present "
                "at the same time"
            )
        for dt in DeviceType:
            if dt == DeviceType.NONE:
                continue
            count = sum(dt in dts for dts in bucket_dts)
            if count > 1 or (base == 1 and count == 1):
                raise VerifyException(
                    "only one of `gang`, `worker`, `vector`, `seq` can be "
                    f"present at the same time for device_type `{dt.value}`"
                )


class _GlobalCtorDtorOperation(IRDLOperation, ABC):
    """Shared shape for `acc.global_ctor` / `acc.global_dtor`.

    Mirrors upstream's `OpenACC_GlobalConstructorOp` / `OpenACC_GlobalDestructorOp`
    — both are module-level `IsolatedFromAbove` + `Symbol` ops carrying just a
    `sym_name` and an unrestricted region. Concrete leaves override only `name`.
    """

    sym_name = prop_def(SymbolNameConstraint())
    region = region_def()

    assembly_format = "$sym_name $region attr-dict-with-keyword"

    def __init__(self, *, sym_name: StringAttr | str, region: Region) -> None:
        sym_name_attr: StringAttr = (
            StringAttr(sym_name) if isinstance(sym_name, str) else sym_name
        )
        super().__init__(
            properties={"sym_name": sym_name_attr},
            regions=[region],
        )


@irdl_op_definition
class GlobalConstructorOp(_GlobalCtorDtorOperation):
    """
    Implementation of upstream acc.global_ctor — captures OpenACC actions
    (e.g. `declare create`) to apply to globals at the entry to the implicit
    data region. Module-level, isolated, named via Symbol.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accglobal_ctor-accglobalconstructorop).
    """

    name = "acc.global_ctor"

    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface())


@irdl_op_definition
class GlobalDestructorOp(_GlobalCtorDtorOperation):
    """
    Implementation of upstream acc.global_dtor — captures OpenACC actions
    (e.g. matching `delete` for a `declare create`) to apply to globals at the
    exit from the implicit data region. Module-level, isolated, named via Symbol.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accglobal_dtor-accglobaldestructorop).
    """

    name = "acc.global_dtor"

    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface())


@irdl_op_definition
class TerminatorOp(IRDLOperation):
    """
    Implementation of upstream acc.terminator. Generic, value-less terminator
    used by OpenACC region ops whose bodies do not return a value.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accterminator-accterminatorop).
    """

    name = "acc.terminator"

    assembly_format = "attr-dict"

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            NoMemoryEffect(),
            HasParent(
                KernelsOp,
                DataOp,
                HostDataOp,
                GlobalConstructorOp,
                GlobalDestructorOp,
                AtomicCaptureOp,
            ),
        )
    )

    def __init__(self) -> None:
        super().__init__()


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    """
    Implementation of upstream acc.yield.
    See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/#accyield-accyieldop).
    """

    name = "acc.yield"

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            NoMemoryEffect(),
            HasParent(
                ParallelOp,
                SerialOp,
                LoopOp,
                PrivateRecipeOp,
                FirstprivateRecipeOp,
                ReductionRecipeOp,
                AtomicUpdateOp,
            ),
        )
    )


ACC = Dialect(
    "acc",
    [
        ParallelOp,
        SerialOp,
        KernelsOp,
        KernelEnvironmentOp,
        LoopOp,
        DataOp,
        HostDataOp,
        DataBoundsOp,
        GetLowerboundOp,
        GetUpperboundOp,
        GetStrideOp,
        GetExtentOp,
        CopyinOp,
        CreateOp,
        PresentOp,
        NoCreateOp,
        AttachOp,
        DevicePtrOp,
        UseDeviceOp,
        CacheOp,
        DeclareDeviceResidentOp,
        DeclareLinkOp,
        GetDevicePtrOp,
        UpdateDeviceOp,
        PrivateOp,
        FirstprivateOp,
        FirstprivateMapOp,
        ReductionOp,
        CopyoutOp,
        UpdateHostOp,
        DeleteOp,
        DetachOp,
        EnterDataOp,
        ExitDataOp,
        UpdateOp,
        AtomicReadOp,
        AtomicWriteOp,
        AtomicUpdateOp,
        AtomicCaptureOp,
        DeclareEnterOp,
        DeclareExitOp,
        DeclareOp,
        PrivateRecipeOp,
        FirstprivateRecipeOp,
        ReductionRecipeOp,
        InitOp,
        ShutdownOp,
        SetOp,
        WaitOp,
        RoutineOp,
        GlobalConstructorOp,
        GlobalDestructorOp,
        TerminatorOp,
        YieldOp,
    ],
    [
        DeviceTypeAttr,
        ClauseDefaultValueAttr,
        DataClauseAttr,
        DataClauseModifierAttr,
        VariableTypeCategoryAttr,
        ReductionOpKindAttr,
        GangArgTypeAttr,
        CombinedConstructsTypeAttr,
        DataBoundsType,
        DeclareTokenType,
    ],
)
