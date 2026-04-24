"""
The OpenACC (acc) dialect that models the OpenACC programming model in MLIR.

OpenACC is a directive-based programming model for accelerating applications
on heterogeneous systems. This dialect exposes compute constructs, data
constructs, loops, and the associated clauses so that host and accelerator
code can be represented, analysed, and lowered to target-specific runtimes.

See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/).
"""

from collections.abc import Sequence
from typing import cast

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
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
    opt_operand_def,
    opt_prop_def,
    region_def,
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
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)


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


def _array_attr_data(attr: Attribute | None) -> tuple[Attribute, ...]:
    """Return the inner tuple of an `ArrayAttr`, or `()` for anything else."""
    if not isinstance(attr, ArrayAttr):
        return ()
    return cast("tuple[Attribute, ...]", attr.data)  # pyright: ignore[reportUnknownMemberType]


def _is_only_device_type_none(dts: Attribute | None) -> bool:
    """Upstream `hasOnlyDeviceTypeNone`: True iff `dts` is exactly `[#none]`."""
    values = _array_attr_data(dts)
    if len(values) != 1:
        return False
    only = values[0]
    return isinstance(only, DeviceTypeAttr) and only.data == DeviceType.NONE


def _parse_device_type_attr(parser: Parser) -> DeviceTypeAttr:
    """Parse a single `#acc.device_type<...>` attribute."""
    attr = parser.parse_attribute()
    if not isinstance(attr, DeviceTypeAttr):
        parser.raise_error("expected #acc.device_type attribute")
    return attr


def _parse_operand_with_optional_dt(
    parser: Parser,
) -> tuple[UnresolvedOperand, Attribute, DeviceTypeAttr]:
    """Parse `%v : <type> ( `[` #acc.device_type<...> `]` )?`.

    Missing device-type defaults to `#acc.device_type<none>` to match upstream.
    """
    operand = parser.parse_unresolved_operand()
    parser.parse_punctuation(":")
    ty = parser.parse_type()
    if parser.parse_optional_punctuation("["):
        dt = _parse_device_type_attr(parser)
        parser.parse_punctuation("]")
    else:
        dt = DeviceTypeAttr(DeviceType.NONE)
    return operand, ty, dt


def _print_device_type_suffix(printer: Printer, dt: Attribute) -> None:
    """Print ` [#acc.device_type<...>]` unless the device type is `#none`."""
    if isinstance(dt, DeviceTypeAttr) and dt.data == DeviceType.NONE:
        return
    printer.print_string(" [")
    printer.print_attribute(dt)
    printer.print_string("]")


def _print_device_type_list(printer: Printer, dts: Sequence[Attribute]) -> None:
    """Print `[#acc.device_type<a>, #acc.device_type<b>, ...]`."""
    printer.print_string("[")
    printer.print_list(dts, printer.print_attribute)
    printer.print_string("]")


def _print_operand_with_dt(printer: Printer, operand: SSAValue, dt: Attribute) -> None:
    """Print `%v : <type> ( [#acc.device_type<...>] )?`."""
    printer.print_ssa_value(operand)
    printer.print_string(" : ")
    printer.print_attribute(operand.type)
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
        operands = self.operands.get(op)
        return bool(operands)

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        operands: list[UnresolvedOperand] = []
        types: list[Attribute] = []
        device_types: list[Attribute] = []
        while True:
            o, t, dt = _parse_operand_with_optional_dt(parser)
            operands.append(o)
            types.append(t)
            device_types.append(dt)
            if not parser.parse_optional_punctuation(","):
                break
        self.operands.set(state, tuple(operands))
        self.operand_types.set(state, tuple(types))
        self.device_types.set(state, ArrayAttr(device_types))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        if not operands:
            return
        dts = _array_attr_data(self.device_types.get(op))
        if not dts:
            dts = (DeviceTypeAttr(DeviceType.NONE),) * len(operands)
        for i, (value, dt) in enumerate(zip(operands, dts)):
            if i:
                printer.print_string(", ")
            _print_operand_with_dt(printer, value, dt)
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class DeviceTypeOperandsWithKeywordOnly(CustomDirective):
    """Port of upstream `custom<DeviceTypeOperandsWithKeywordOnly>`.

    Follows a bare `async` keyword in the format. The directive owns the
    optional surrounding parentheses. Syntax options (after the keyword):
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
        operands = self.operands.get(op)
        if operands:
            return True
        keyword_only = self.keyword_only.get(op)
        device_types = self.device_types.get(op)
        return keyword_only is not None or device_types is not None

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if not parser.parse_optional_punctuation("("):
            # bare keyword form → keyword-only list = [#none]
            self.keyword_only.set(state, ArrayAttr([DeviceTypeAttr(DeviceType.NONE)]))
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        needs_comma_before_operands = False
        if parser.parse_optional_punctuation("["):
            keyword_only_attrs: list[Attribute] = [_parse_device_type_attr(parser)]
            while parser.parse_optional_punctuation(","):
                keyword_only_attrs.append(_parse_device_type_attr(parser))
            parser.parse_punctuation("]")
            self.keyword_only.set(state, ArrayAttr(keyword_only_attrs))
            needs_comma_before_operands = True

        if parser.parse_optional_punctuation(")"):
            # keyword-only list with no operands
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        if needs_comma_before_operands:
            parser.parse_punctuation(",")

        operands: list[UnresolvedOperand] = []
        types: list[Attribute] = []
        device_types: list[Attribute] = []
        while True:
            o, t, dt = _parse_operand_with_optional_dt(parser)
            operands.append(o)
            types.append(t)
            device_types.append(dt)
            if not parser.parse_optional_punctuation(","):
                break
        parser.parse_punctuation(")")
        self.operands.set(state, tuple(operands))
        self.operand_types.set(state, tuple(types))
        self.device_types.set(state, ArrayAttr(device_types))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        keyword_only = self.keyword_only.get(op)

        # bare keyword (no parens): operands empty and keyword-only is [#none].
        if not operands and _is_only_device_type_none(keyword_only):
            return

        kwords = _array_attr_data(keyword_only)
        dts = _array_attr_data(self.device_types.get(op))
        if operands and not dts:
            dts = (DeviceTypeAttr(DeviceType.NONE),) * len(operands)

        printer.print_string("(")
        if kwords:
            _print_device_type_list(printer, kwords)
            if operands:
                printer.print_string(", ")
        for i, (value, dt) in enumerate(zip(operands, dts)):
            if i:
                printer.print_string(", ")
            _print_operand_with_dt(printer, value, dt)
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
        operands: list[UnresolvedOperand] = []
        types: list[Attribute] = []
        device_types: list[Attribute] = []
        seg: list[int] = []
        while True:
            parser.parse_punctuation("{")
            group_start = len(operands)
            while True:
                o = parser.parse_unresolved_operand()
                parser.parse_punctuation(":")
                t = parser.parse_type()
                operands.append(o)
                types.append(t)
                if not parser.parse_optional_punctuation(","):
                    break
            seg.append(len(operands) - group_start)
            parser.parse_punctuation("}")
            if parser.parse_optional_punctuation("["):
                device_types.append(_parse_device_type_attr(parser))
                parser.parse_punctuation("]")
            else:
                device_types.append(DeviceTypeAttr(DeviceType.NONE))
            if not parser.parse_optional_punctuation(","):
                break
        self.operands.set(state, tuple(operands))
        self.operand_types.set(state, tuple(types))
        self.device_types.set(state, ArrayAttr(device_types))
        self.segments.set(state, DenseArrayBase.from_list(i32, seg))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        if not operands:
            return
        dts = _array_attr_data(self.device_types.get(op))
        if not dts:
            dts = (DeviceTypeAttr(DeviceType.NONE),)
        segments_attr = self.segments.get(op)
        seg_values: Sequence[int] = (
            tuple(segments_attr.get_values())
            if isinstance(segments_attr, DenseArrayBase)
            else (len(operands),)
        )
        idx = 0
        for group_idx, (size, dt) in enumerate(zip(seg_values, dts)):
            if group_idx:
                printer.print_string(", ")
            printer.print_string("{")
            for i in range(size):
                if i:
                    printer.print_string(", ")
                value = operands[idx]
                printer.print_ssa_value(value)
                printer.print_string(" : ")
                printer.print_attribute(value.type)
                idx += 1
            printer.print_string("}")
            _print_device_type_suffix(printer, dt)
        state.should_emit_space = True
        state.last_was_punctuation = False


@irdl_custom_directive
class WaitClause(CustomDirective):
    """Port of upstream `custom<WaitClause>`.

    Extends `NumGangs`-style groups with an optional leading keyword-only
    device-type list and an optional `devnum:` marker per group. Follows a
    bare `wait` keyword in the format; the directive owns the surrounding
    parentheses (if any).
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
        if self.operands.get(op):
            return True
        return (
            self.keyword_only.get(op) is not None
            or self.device_types.get(op) is not None
        )

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set(state, ())
        self.operand_types.set(state, ())

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if not parser.parse_optional_punctuation("("):
            # bare `wait` → keyword-only = [#none]
            self.keyword_only.set(state, ArrayAttr([DeviceTypeAttr(DeviceType.NONE)]))
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        needs_comma_before_operands = False
        if parser.parse_optional_punctuation("["):
            keyword_only_attrs: list[Attribute] = [_parse_device_type_attr(parser)]
            while parser.parse_optional_punctuation(","):
                keyword_only_attrs.append(_parse_device_type_attr(parser))
            parser.parse_punctuation("]")
            self.keyword_only.set(state, ArrayAttr(keyword_only_attrs))
            needs_comma_before_operands = True

        if parser.parse_optional_punctuation(")"):
            # keyword-only list with no braced groups
            self.operands.set(state, ())
            self.operand_types.set(state, ())
            return True

        if needs_comma_before_operands:
            parser.parse_punctuation(",")

        operands: list[UnresolvedOperand] = []
        types: list[Attribute] = []
        device_types: list[Attribute] = []
        seg: list[int] = []
        has_devnum: list[Attribute] = []
        while True:
            parser.parse_punctuation("{")
            group_start = len(operands)
            if parser.parse_optional_keyword("devnum"):
                parser.parse_punctuation(":")
                has_devnum.append(BoolAttr.from_bool(True))
            else:
                has_devnum.append(BoolAttr.from_bool(False))
            while True:
                o = parser.parse_unresolved_operand()
                parser.parse_punctuation(":")
                t = parser.parse_type()
                operands.append(o)
                types.append(t)
                if not parser.parse_optional_punctuation(","):
                    break
            seg.append(len(operands) - group_start)
            parser.parse_punctuation("}")
            if parser.parse_optional_punctuation("["):
                device_types.append(_parse_device_type_attr(parser))
                parser.parse_punctuation("]")
            else:
                device_types.append(DeviceTypeAttr(DeviceType.NONE))
            if not parser.parse_optional_punctuation(","):
                break
        parser.parse_punctuation(")")
        self.operands.set(state, tuple(operands))
        self.operand_types.set(state, tuple(types))
        self.device_types.set(state, ArrayAttr(device_types))
        self.segments.set(state, DenseArrayBase.from_list(i32, seg))
        self.has_devnum.set(state, ArrayAttr(has_devnum))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operands = self.operands.get(op)
        keyword_only = self.keyword_only.get(op)
        segments_attr = self.segments.get(op)

        if not operands and _is_only_device_type_none(keyword_only):
            return

        kwords = _array_attr_data(keyword_only)

        printer.print_string("(")
        if kwords:
            _print_device_type_list(printer, kwords)

        if operands:
            dts = _array_attr_data(self.device_types.get(op))
            if not dts:
                dts = (DeviceTypeAttr(DeviceType.NONE),)
            seg_values: Sequence[int] = (
                tuple(segments_attr.get_values())
                if isinstance(segments_attr, DenseArrayBase)
                else (len(operands),)
            )
            devnum_flags = _array_attr_data(self.has_devnum.get(op))
            if kwords:
                printer.print_string(", ")
            idx = 0
            for group_idx, (size, dt) in enumerate(zip(seg_values, dts)):
                if group_idx:
                    printer.print_string(", ")
                printer.print_string("{")
                devnum_attr = (
                    devnum_flags[group_idx] if group_idx < len(devnum_flags) else None
                )
                if isinstance(devnum_attr, IntegerAttr) and bool(
                    devnum_attr.value.data
                ):
                    printer.print_string("devnum: ")
                for i in range(size):
                    if i:
                        printer.print_string(", ")
                    value = operands[idx]
                    printer.print_ssa_value(value)
                    printer.print_string(" : ")
                    printer.print_attribute(value.type)
                    idx += 1
                printer.print_string("}")
                _print_device_type_suffix(printer, dt)
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
    wait_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="waitOperandsSegments"
    )
    wait_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="waitOperandsDeviceType"
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
        DeviceTypeOperandsWithKeywordOnly,
        NumGangs,
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
        wait_operands_segments: DenseArrayBase | None = None,
        wait_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
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
                "waitOperandsSegments": wait_operands_segments,
                "waitOperandsDeviceType": wait_operands_device_type,
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
            HasParent(ParallelOp),
        )
    )


ACC = Dialect(
    "acc",
    [
        ParallelOp,
        YieldOp,
    ],
    [
        DeviceTypeAttr,
        ClauseDefaultValueAttr,
    ],
)
