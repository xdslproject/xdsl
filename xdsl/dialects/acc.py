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


def _device_type_none(*, n: int = 1) -> ArrayAttr[DeviceTypeAttr]:
    return ArrayAttr([DeviceTypeAttr(DeviceType.NONE)] * n)


def _is_only_device_type_none(attr: Attribute | None) -> bool:
    if not isinstance(attr, ArrayAttr) or len(attr.data) != 1:
        return False
    entry = attr.data[0]
    return isinstance(entry, DeviceTypeAttr) and entry.data == DeviceType.NONE


def _parse_device_type_bracket_list(parser: Parser) -> list[DeviceTypeAttr]:
    """Parses `[#acc.device_type<..>, ...]`. Assumes the opening `[` has been seen."""
    items = parser.parse_comma_separated_list(
        parser.Delimiter.NONE,
        lambda: cast(DeviceTypeAttr, parser.parse_attribute()),
    )
    parser.parse_punctuation("]")
    return items


def _print_device_type_bracket_list(
    printer: Printer, device_types: Sequence[DeviceTypeAttr]
) -> None:
    printer.print_string("[")
    printer.print_list(device_types, printer.print_attribute)
    printer.print_string("]")


@irdl_custom_directive
class AsyncClause(CustomDirective):
    """
    Custom directive for the `async` clause of acc ops.
    Mirrors MLIR's `custom<DeviceTypeOperandsWithKeywordOnly>` printer/parser.

    Syntax:
      async-clause ::= `async` ( `(` async-body `)` )?
      async-body   ::= (kw-dts)? | (kw-dts `,`)? operand-dt (`,` operand-dt)*
      kw-dts       ::= `[` device-type (`,` device-type)* `]`
      operand-dt   ::= operand `:` type (`[` device-type `]`)?

    A bare `async` keyword means the operation carries `asyncOnly = [none]`.
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    device_types: AttributeVariable  # asyncOperandsDeviceType
    keyword_only: AttributeVariable  # asyncOnly

    def is_present(self, op: IRDLOperation) -> bool:
        if self.operands.is_present(op):
            return True
        dts = self.device_types.get(op)
        if isinstance(dts, ArrayAttr) and len(dts.data) > 0:
            return True
        kw = self.keyword_only.get(op)
        return isinstance(kw, ArrayAttr) and len(kw.data) > 0

    def is_anchorable(self) -> bool:
        return True

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set_empty(state)
        self.operand_types.set_empty(state)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if parser.parse_optional_punctuation("(") is None:
            self.keyword_only.set(state, _device_type_none())
            self.operands.set_empty(state)
            self.operand_types.set_empty(state)
            return True

        keyword_only_dts: list[DeviceTypeAttr] = []
        if parser.parse_optional_punctuation("[") is not None:
            keyword_only_dts = _parse_device_type_bracket_list(parser)
            if parser.parse_optional_punctuation(")") is not None:
                self.keyword_only.set(state, ArrayAttr(keyword_only_dts))
                self.operands.set_empty(state)
                self.operand_types.set_empty(state)
                return True
            parser.parse_punctuation(",")

        if keyword_only_dts:
            self.keyword_only.set(state, ArrayAttr(keyword_only_dts))

        ops: list[UnresolvedOperand] = []
        types: list[Attribute] = []
        dts: list[DeviceTypeAttr] = []
        while True:
            ops.append(parser.parse_unresolved_operand())
            parser.parse_punctuation(":")
            types.append(parser.parse_type())
            if parser.parse_optional_punctuation("[") is not None:
                dts.append(cast(DeviceTypeAttr, parser.parse_attribute()))
                parser.parse_punctuation("]")
            else:
                dts.append(DeviceTypeAttr(DeviceType.NONE))
            if parser.parse_optional_punctuation(",") is None:
                break
        parser.parse_punctuation(")")

        self.operands.set(state, ops)
        self.operand_types.set(state, types)
        self.device_types.set(state, ArrayAttr(dts))
        return True

    def print(
        self, printer: Printer, state: PrintingState, op: IRDLOperation
    ) -> None:
        operands = tuple(self.operands.get(op))
        keyword_only = self.keyword_only.get(op)

        if not operands and _is_only_device_type_none(keyword_only):
            state.should_emit_space = True
            state.last_was_punctuation = False
            return

        operand_types = tuple(self.operand_types.get(op))
        device_types_attr = self.device_types.get(op)
        dts: Sequence[DeviceTypeAttr] = (
            device_types_attr.data
            if isinstance(device_types_attr, ArrayAttr)
            else [DeviceTypeAttr(DeviceType.NONE)] * len(operands)
        )
        kw_dts: Sequence[DeviceTypeAttr] = (
            keyword_only.data if isinstance(keyword_only, ArrayAttr) else ()
        )

        printer.print_string("(")
        if kw_dts:
            _print_device_type_bracket_list(printer, kw_dts)
            if operands:
                printer.print_string(", ")

        for i, (v, t) in enumerate(zip(operands, operand_types)):
            if i:
                printer.print_string(", ")
            printer.print_ssa_value(v)
            printer.print_string(" : ")
            printer.print_attribute(t)
            dt = dts[i] if i < len(dts) else DeviceTypeAttr(DeviceType.NONE)
            if dt.data != DeviceType.NONE:
                _print_device_type_bracket_list(printer, (dt,))

        printer.print_string(")")
        state.last_was_punctuation = True
        state.should_emit_space = True


@irdl_custom_directive
class WaitClause(CustomDirective):
    """
    Custom directive for the `wait` clause of acc ops.
    Mirrors MLIR's `custom<WaitClause>` printer/parser.

    Syntax:
      wait-clause ::= `wait` ( `(` wait-body `)` )?
      wait-body   ::= kw-dts
                    | (kw-dts `,`)? wait-seg (`,` wait-seg)*
      kw-dts      ::= `[` device-type (`,` device-type)* `]`
      wait-seg    ::= `{` (`devnum` `:`)? op-type-list `}` (`[` device-type `]`)?
      op-type-list ::= operand `:` type (`,` operand `:` type)*

    A bare `wait` keyword means the operation carries `waitOnly = [none]`.
    """

    operands: VariadicOperandVariable
    operand_types: TypeDirective
    device_types: AttributeVariable  # waitOperandsDeviceType
    segments: AttributeVariable  # waitOperandsSegments
    has_devnum: AttributeVariable  # hasWaitDevnum
    keyword_only: AttributeVariable  # waitOnly

    def is_present(self, op: IRDLOperation) -> bool:
        if self.operands.is_present(op):
            return True
        dts = self.device_types.get(op)
        if isinstance(dts, ArrayAttr) and len(dts.data) > 0:
            return True
        kw = self.keyword_only.get(op)
        return isinstance(kw, ArrayAttr) and len(kw.data) > 0

    def is_anchorable(self) -> bool:
        return True

    def set_empty(self, state: ParsingState) -> None:
        self.operands.set_empty(state)
        self.operand_types.set_empty(state)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if parser.parse_optional_punctuation("(") is None:
            self.keyword_only.set(state, _device_type_none())
            self.operands.set_empty(state)
            self.operand_types.set_empty(state)
            return True

        keyword_only_dts: list[DeviceTypeAttr] = []
        if parser.parse_optional_punctuation("[") is not None:
            keyword_only_dts = _parse_device_type_bracket_list(parser)
            if parser.parse_optional_punctuation(")") is not None:
                self.keyword_only.set(state, ArrayAttr(keyword_only_dts))
                self.operands.set_empty(state)
                self.operand_types.set_empty(state)
                return True
            parser.parse_punctuation(",")

        if keyword_only_dts:
            self.keyword_only.set(state, ArrayAttr(keyword_only_dts))

        ops: list[UnresolvedOperand] = []
        types: list[Attribute] = []
        dts: list[DeviceTypeAttr] = []
        seg_sizes: list[int] = []
        devnums: list[BoolAttr] = []

        while True:
            parser.parse_punctuation("{")
            has_devnum = parser.parse_optional_keyword("devnum") is not None
            if has_devnum:
                parser.parse_punctuation(":")
            devnums.append(BoolAttr.from_bool(has_devnum))

            seg_start = len(ops)
            while True:
                ops.append(parser.parse_unresolved_operand())
                parser.parse_punctuation(":")
                types.append(parser.parse_type())
                if parser.parse_optional_punctuation(",") is None:
                    break
            seg_sizes.append(len(ops) - seg_start)
            parser.parse_punctuation("}")

            if parser.parse_optional_punctuation("[") is not None:
                dts.append(cast(DeviceTypeAttr, parser.parse_attribute()))
                parser.parse_punctuation("]")
            else:
                dts.append(DeviceTypeAttr(DeviceType.NONE))

            if parser.parse_optional_punctuation(",") is None:
                break
        parser.parse_punctuation(")")

        self.operands.set(state, ops)
        self.operand_types.set(state, types)
        self.device_types.set(state, ArrayAttr(dts))
        self.segments.set(state, DenseArrayBase.from_list(i32, seg_sizes))
        self.has_devnum.set(state, ArrayAttr(devnums))
        return True

    def print(
        self, printer: Printer, state: PrintingState, op: IRDLOperation
    ) -> None:
        operands = tuple(self.operands.get(op))
        keyword_only = self.keyword_only.get(op)

        if not operands and _is_only_device_type_none(keyword_only):
            state.should_emit_space = True
            state.last_was_punctuation = False
            return

        operand_types = tuple(self.operand_types.get(op))
        device_types_attr = self.device_types.get(op)
        segments_attr = self.segments.get(op)
        has_devnum_attr = self.has_devnum.get(op)

        kw_dts: Sequence[DeviceTypeAttr] = (
            keyword_only.data if isinstance(keyword_only, ArrayAttr) else ()
        )

        printer.print_string("(")
        if kw_dts:
            _print_device_type_bracket_list(printer, kw_dts)
            if operands:
                printer.print_string(", ")

        if operands:
            dts: Sequence[DeviceTypeAttr] = (
                device_types_attr.data
                if isinstance(device_types_attr, ArrayAttr)
                else ()
            )
            seg_sizes: Sequence[int] = (
                tuple(segments_attr.iter_values())
                if isinstance(segments_attr, DenseArrayBase)
                else (len(operands),)
            )
            devnum_flags: Sequence[BoolAttr] = (
                has_devnum_attr.data
                if isinstance(has_devnum_attr, ArrayAttr)
                else ()
            )

            op_idx = 0
            for seg_idx, seg_size in enumerate(seg_sizes):
                if seg_idx:
                    printer.print_string(", ")
                printer.print_string("{")
                if seg_idx < len(devnum_flags) and bool(
                    devnum_flags[seg_idx].value.data
                ):
                    printer.print_string("devnum: ")
                for i in range(seg_size):
                    if i:
                        printer.print_string(", ")
                    printer.print_ssa_value(operands[op_idx])
                    printer.print_string(" : ")
                    printer.print_attribute(operand_types[op_idx])
                    op_idx += 1
                printer.print_string("}")
                dt = (
                    dts[seg_idx]
                    if seg_idx < len(dts)
                    else DeviceTypeAttr(DeviceType.NONE)
                )
                if dt.data != DeviceType.NONE:
                    _print_device_type_bracket_list(printer, (dt,))

        printer.print_string(")")
        state.last_was_punctuation = True
        state.should_emit_space = True


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

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

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
    wait_operands_segments = opt_prop_def(
        DenseArrayBase.constr(IntegerType(32)), prop_name="waitOperandsSegments"
    )
    wait_operands_device_type = opt_prop_def(
        ArrayAttr[DeviceTypeAttr], prop_name="waitOperandsDeviceType"
    )
    has_wait_devnum = opt_prop_def(ArrayAttr[BoolAttr], prop_name="hasWaitDevnum")
    wait_only = opt_prop_def(ArrayAttr[DeviceTypeAttr], prop_name="waitOnly")
    self_attr = opt_prop_def(UnitAttr, prop_name="selfAttr")
    default_attr = opt_prop_def(ClauseDefaultValueAttr, prop_name="defaultAttr")
    combined = opt_prop_def(UnitAttr)

    region = region_def("single_block")

    irdl_options = (AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict())

    assembly_format = (
        "(`combined` `(` `loop` `)` $combined^)?"
        " (`dataOperands` `(` $data_clause_operands^ `:`"
        " type($data_clause_operands) `)`)?"
        " (`async` `` custom<AsyncClause>($async_operands,"
        " type($async_operands), $asyncOperandsDeviceType, $asyncOnly)^)?"
        " (`firstprivate` `(` $firstprivate_operands^ `:`"
        " type($firstprivate_operands) `)`)?"
        " (`private` `(` $private_operands^ `:` type($private_operands) `)`)?"
        " (`wait` `` custom<WaitClause>($wait_operands, type($wait_operands),"
        " $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,"
        " $waitOnly)^)?"
        " (`self` `(` $self_cond^ `)`)?"
        " (`if` `(` $if_cond^ `)`)?"
        " (`reduction` `(` $reduction_operands^ `:`"
        " type($reduction_operands) `)`)?"
        " $region attr-dict-with-keyword"
    )

    custom_directives = (AsyncClause, WaitClause)

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
        wait_operands_segments: DenseArrayBase | None = None,
        wait_operands_device_type: ArrayAttr[DeviceTypeAttr] | None = None,
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
                "waitOperandsSegments": wait_operands_segments,
                "waitOperandsDeviceType": wait_operands_device_type,
                "hasWaitDevnum": has_wait_devnum,
                "waitOnly": wait_only,
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
            HasParent(ParallelOp, SerialOp),
        )
    )


ACC = Dialect(
    "acc",
    [
        ParallelOp,
        SerialOp,
        YieldOp,
    ],
    [
        DeviceTypeAttr,
        ClauseDefaultValueAttr,
    ],
)
