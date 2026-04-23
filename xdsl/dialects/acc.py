"""
The OpenACC (acc) dialect that models the OpenACC programming model in MLIR.

OpenACC is a directive-based programming model for accelerating applications
on heterogeneous systems. This dialect exposes compute constructs, data
constructs, loops, and the associated clauses so that host and accelerator
code can be represented, analysed, and lowered to target-specific runtimes.

See external [documentation](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/).
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    I1,
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    IndexType,
    IntegerType,
    UnitAttr,
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
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
    opt_operand_def,
    opt_prop_def,
    region_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
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
