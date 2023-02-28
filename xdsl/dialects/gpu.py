from __future__ import annotations

from typing import Annotated, Generic, Type, TypeVar

from xdsl.dialects.builtin import IndexType, StringAttr, SymbolRefAttr, i32
from xdsl.ir import Attribute, OpResult, Operation, ParametrizedAttribute, SSAValue
from xdsl.irdl import Operand, ParameterDef, irdl_op_definition, irdl_attr_definition, SingleBlockRegion, OpAttr
from xdsl.parser import BaseParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class _AllReduceOperationAttr(ParametrizedAttribute):
    name = "all_reduce_op"

    param: ParameterDef[StringAttr]

    def print_parameters(self, printer: Printer) -> None:
        printer.print(f"all_reduce_op {self.param.data}")


@irdl_attr_definition
class _DimensionAttr(ParametrizedAttribute):
    name = "dim"

    param: ParameterDef[StringAttr]

    def print_parameters(self, printer: Printer) -> None:
        printer.print(f"dim {self.param.data}")


T = TypeVar('T',
            bound=_AllReduceOperationAttr | _DimensionAttr,
            covariant=True)


@irdl_attr_definition
class _GPUAttr(ParametrizedAttribute, Generic[T]):
    name = "gpu"

    value: ParameterDef[T]

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        parser.parse_characters(
            "<",
            f"Expected <. gpu attributes currently have the #gpu<name value> syntax."
        )
        ntok = parser.tokenizer.next_token()

        if ntok.text == "dim":
            attrtype = _DimensionAttr
            vtok = parser.tokenizer.next_token()
            if vtok.text not in ["x", "y", "z"]:
                parser.raise_error(
                    f"Unexpected dim {vtok.text}. A gpu dim can only be x, y, or z",
                    vtok)

        elif ntok.text == "all_reduce_op":
            attrtype = _AllReduceOperationAttr
            vtok = parser.tokenizer.next_token()
            if vtok.text not in [
                    "add", "and", "max", "min", "mul", "or", "xor"
            ]:
                parser.raise_error(
                    f"Unexpected op {vtok.text}. A gpu all_reduce_op can only be add, "
                    "and, max, min, mul, or, or xor ", vtok)
        else:
            parser.raise_error(
                f"Unexpected token {ntok.text}. Expected dim or all_reduce_op",
                ntok)
        parser.parse_characters(
            ">",
            f"Expected >. gpu attributes currently have the #gpu<name value> syntax."
        )
        return [attrtype([StringAttr.from_str(vtok.text)])]

    @classmethod
    def from_op(cls: Type[_GPUAttr[_AllReduceOperationAttr]],
                value: str) -> AllReduceOperationAttr:
        return AllReduceOperationAttr(
            [_AllReduceOperationAttr([StringAttr(value)])])

    @property
    def data(self) -> str:
        return self.value.param.data

    @classmethod
    def from_dimension(cls: Type[_GPUAttr[_DimensionAttr]],
                       value: str) -> DimensionAttr:
        return DimensionAttr([_DimensionAttr([StringAttr(value)])])

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        self.value.print_parameters(printer)
        printer.print_string(">")


DimensionAttr = _GPUAttr[_DimensionAttr]
AllReduceOperationAttr = _GPUAttr[_AllReduceOperationAttr]


@irdl_op_definition
class BarrierOp(Operation):
    name = "gpu.barrier"

    @staticmethod
    def get() -> BarrierOp:
        return BarrierOp.build()


@irdl_op_definition
class BlockDimOp(Operation):
    name = "gpu.block_dim"
    dimension: OpAttr[DimensionAttr]
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get(dim: DimensionAttr) -> BlockDimOp:
        return BlockDimOp.build(result_types=[IndexType()],
                                attributes={"dimension": dim})


@irdl_op_definition
class BlockIdOp(Operation):
    name = "gpu.block_id"
    dimension: OpAttr[DimensionAttr]
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get(dim: DimensionAttr) -> BlockIdOp:
        return BlockIdOp.build(result_types=[IndexType()],
                               attributes={"dimension": dim})


@irdl_op_definition
class ModuleOp(Operation):
    name = "gpu.module"

    body: SingleBlockRegion
    sym_name: OpAttr[StringAttr]

    @staticmethod
    def get(name: SymbolRefAttr, ops: list[Operation]) -> ModuleOp:
        op = ModuleOp.build(attributes={"sym_name": name}, regions=[ops])
        return op

    def verify_(self):
        if (len(self.body.ops) == 0
                or not isinstance(self.body.ops[-1], ModuleEndOp)):
            raise VerifyException("gpu.module must end with gpu.module_end")


@irdl_op_definition
class GlobalIdOp(Operation):
    name = "gpu.global_id"
    dimension: OpAttr[DimensionAttr]
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get(dim: DimensionAttr) -> GlobalIdOp:
        return GlobalIdOp.build(result_types=[IndexType()],
                                attributes={"dimension": dim})


@irdl_op_definition
class GridDimOp(Operation):
    name = "gpu.grid_dim"
    dimension: OpAttr[DimensionAttr]
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get(dim: DimensionAttr) -> GridDimOp:
        return GridDimOp.build(result_types=[IndexType()],
                               attributes={"dimension": dim})


@irdl_op_definition
class LaneIdOp(Operation):
    name = "gpu.lane_id"
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get() -> LaneIdOp:
        return LaneIdOp.build(result_types=[IndexType()])


@irdl_op_definition
class ModuleEndOp(Operation):
    name = "gpu.module_end"

    @staticmethod
    def get() -> ModuleEndOp:
        return ModuleEndOp.build()


@irdl_op_definition
class NumSubgroupsOp(Operation):
    name = "gpu.num_subgroups"
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get() -> NumSubgroupsOp:
        return NumSubgroupsOp.build(result_types=[IndexType()])


@irdl_op_definition
class SetDefaultDeviceOp(Operation):
    name = "gpu.set_default_device"
    devIndex: Annotated[Operand, i32]

    @staticmethod
    def get(devIndex: SSAValue | Operation) -> SetDefaultDeviceOp:
        return SetDefaultDeviceOp.build(operands=[SSAValue.get(devIndex)])


@irdl_op_definition
class SubgroupIdOp(Operation):
    name = "gpu.subgroup_id"
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get() -> SubgroupIdOp:
        return SubgroupIdOp.build(result_types=[IndexType()])


@irdl_op_definition
class SubgroupSizeOp(Operation):
    name = "gpu.subgroup_size"
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get() -> SubgroupSizeOp:
        return SubgroupSizeOp.build(result_types=[IndexType()])


@irdl_op_definition
class ThreadIdOp(Operation):
    name = "gpu.thread_id"
    dimension: OpAttr[DimensionAttr]
    result: Annotated[OpResult, IndexType]

    @staticmethod
    def get(dim: DimensionAttr) -> ThreadIdOp:
        return ThreadIdOp.build(result_types=[IndexType()],
                                attributes={"dimension": dim})
