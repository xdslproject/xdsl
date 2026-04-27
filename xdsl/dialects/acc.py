"""
The OpenACC (acc) dialect that models the OpenACC programming model in MLIR.

OpenACC is a directive-based programming model for accelerating applications
on heterogeneous systems. This dialect exposes compute constructs, data
constructs, loops, and the associated clauses so that host and accelerator
code can be represented, analysed, and lowered to target-specific runtimes.

See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/).
"""

from abc import ABC
from collections.abc import Sequence

from xdsl.dialects.builtin import (
    I1,
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    UnitAttr,
    i32,
)
from xdsl.dialects.utils import AbstractYieldOperation
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
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.irdl.declarative_assembly_format import (
    AttributeVariable,
    CustomDirective,
    ParsingState,
    PrintingState,
    TypeDirective,
    VariadicOperandVariable,
    irdl_custom_directive,
)
from xdsl.parser import AttrParser, Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoMemoryEffect,
    NoTerminator,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
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
class DataBoundsType(ParametrizedAttribute, TypeAttribute):
    """
    Type used for `acc.bounds` results. Holds normalized bounds information
    for an `acc` data clause; consumed by the data-clause ops via their
    variadic `bounds` operand and by the `acc.get_*bound`/`get_extent`/
    `get_stride` accessor ops.
    See upstream `!acc.data_bounds_ty`.
    """

    name = "acc.data_bounds_ty"


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
        if not parser.parse_optional_punctuation("("):
            self.keyword_only.set(state, ArrayAttr([DeviceTypeAttr(DeviceType.NONE)]))
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        needs_comma_before_operands = False
        if parser.parse_optional_punctuation("["):
            kw_only = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, lambda: _parse_device_type_attr(parser)
            )
            parser.parse_punctuation("]")
            self.keyword_only.set(state, ArrayAttr(kw_only))
            needs_comma_before_operands = True

        if parser.parse_optional_punctuation(")"):
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        if needs_comma_before_operands:
            parser.parse_punctuation(",")

        triples = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, lambda: _parse_operand_with_dt(parser)
        )
        parser.parse_punctuation(")")
        operands, types, device_types = zip(*triples)
        self.operands.set(state, operands)
        self.operand_types.set(state, types)
        self.device_types.set(state, ArrayAttr(device_types))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        keyword_only = self.keyword_only.get(op)

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
                if isa(attr := self.device_types.get(op), ArrayAttr)
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
        groups = [_parse_num_gangs_group(parser)]
        while parser.parse_optional_punctuation(","):
            groups.append(_parse_num_gangs_group(parser))
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


@irdl_custom_directive
class WaitClause(CustomDirective):
    """Port of upstream `custom<WaitClause>`.

    Follows a bare `wait` keyword in the format. The directive owns the
    optional surrounding parentheses. Syntax options after the keyword:
      bare                                                → keyword-only = [#none]
      `(` `[` dts `]` `,` group (`,` group)* `)`          → kw-only + operand groups
      `(` group (`,` group)* `)`                          → operand groups only
    where each group is `{ (`devnum:`)? %v : type, ... } (`[` dt `]`)?`.
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
        if not parser.parse_optional_punctuation("("):
            self.keyword_only.set(state, ArrayAttr([DeviceTypeAttr(DeviceType.NONE)]))
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        if parser.parse_optional_punctuation("["):
            kw_only = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, lambda: _parse_device_type_attr(parser)
            )
            parser.parse_punctuation("]")
            self.keyword_only.set(state, ArrayAttr(kw_only))
            parser.parse_punctuation(",")

        groups = [_parse_wait_group(parser)]
        while parser.parse_optional_punctuation(","):
            groups.append(_parse_wait_group(parser))
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

        self.operands.set(state, tuple(all_operands))
        self.operand_types.set(state, tuple(all_types))
        self.device_types.set(state, ArrayAttr(dts))
        self.segments.set(state, DenseArrayBase.from_list(i32, segs))
        self.has_devnum.set(state, ArrayAttr(devnum_flags))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        keyword_only = self.keyword_only.get(op)

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
                if isa(attr := self.device_types.get(op), ArrayAttr)
                else (DeviceTypeAttr(DeviceType.NONE),)
            )
            seg_values: Sequence[int] = (
                segments.get_values()
                if isinstance(segments := self.segments.get(op), DenseArrayBase)
                else (len(operands),)
            )
            devnum_flags: Sequence[Attribute] = (
                devnum_attrs.data
                if isa(devnum_attrs := self.has_devnum.get(op), ArrayAttr)
                else ()
            )
            _print_groups(printer, operands, dts, seg_values, devnum_flags=devnum_flags)
        printer.print_string(")")
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

    # Upstream `acc.kernels` body uses `acc.terminator` rather than `acc.yield`
    # That will be supported later, until that op exists we accept any (or empty)
    # body via NoTerminator, mirroring upstream's `AnyRegion` modeling.
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
            NoTerminator(),
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
            HasParent(ParallelOp, SerialOp),
        )
    )


ACC = Dialect(
    "acc",
    [
        ParallelOp,
        SerialOp,
        KernelsOp,
        DataBoundsOp,
        GetLowerboundOp,
        GetUpperboundOp,
        GetStrideOp,
        GetExtentOp,
        YieldOp,
    ],
    [
        DeviceTypeAttr,
        ClauseDefaultValueAttr,
        DataBoundsType,
    ],
)
