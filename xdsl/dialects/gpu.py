from __future__ import annotations

from collections.abc import Sequence
from enum import auto
from typing import TypeVar

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    FunctionType,
    IndexType,
    IntegerAttr,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    i32,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    EnumAttribute,
    OpaqueSyntaxAttribute,
    Operation,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    OptOperand,
    OptOpResult,
    ParameterDef,
    VarOperand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    SingleBlockImplicitTerminator,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class AsyncTokenType(ParametrizedAttribute, TypeAttribute):
    name = "gpu.async.token"


class AllReduceOpEnum(StrEnum):
    Add = auto()
    And = auto()
    Max = auto()
    Min = auto()
    Mul = auto()
    Or = auto()
    Xor = auto()


class DimensionEnum(StrEnum):
    X = auto()
    Y = auto()
    Z = auto()


class ProcessorEnum(StrEnum):
    Sequential = auto()
    Block_X = auto()
    Block_Y = auto()
    Block_Z = auto()
    Thread_X = auto()
    Thread_Y = auto()
    Thread_Z = auto()


@irdl_attr_definition
class AllReduceOpAttr(EnumAttribute[AllReduceOpEnum], OpaqueSyntaxAttribute):
    name = "gpu.all_reduce_op"


@irdl_attr_definition
class DimensionAttr(EnumAttribute[DimensionEnum], OpaqueSyntaxAttribute):
    name = "gpu.dim"


@irdl_attr_definition
class ProcessorAttr(EnumAttribute[ProcessorEnum], OpaqueSyntaxAttribute):
    name = "gpu.processor"


@irdl_attr_definition
class LoopDimMapAttr(ParametrizedAttribute):
    name = "gpu.loop_dim_map"

    processor: ParameterDef[ProcessorAttr]
    map: ParameterDef[AffineMapAttr]
    bound: ParameterDef[AffineMapAttr]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print("processor = ")
            printer.print(self.processor.data)
            printer.print(", map = ")
            printer.print(self.map.data)
            printer.print(", bound = ")
            printer.print(self.bound.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            parser.parse_keyword("processor")
            parser.parse_punctuation("=")
            proc = ProcessorAttr.parse_parameter(parser)
            processor = ProcessorAttr(proc)
            parser.parse_punctuation(",")
            parser.parse_keyword("map")
            parser.parse_punctuation("=")
            map = AffineMapAttr(parser.parse_affine_map())
            parser.parse_punctuation(",")
            parser.parse_keyword("bound")
            parser.parse_punctuation("=")
            bound = AffineMapAttr(parser.parse_affine_map())
        return [processor, map, bound]


_Element = TypeVar("_Element", bound=Attribute, covariant=True)


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "gpu.alloc"
    hostShared: UnitAttr | None = opt_attr_def(UnitAttr)
    asyncDependencies: VarOperand = var_operand_def(AsyncTokenType)
    dynamicSizes: VarOperand = var_operand_def(IndexType)
    symbolOperands: VarOperand = var_operand_def(IndexType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    result: OpResult = result_def(memref.MemRefType[Attribute])
    asyncToken: OptOpResult = opt_result_def(AsyncTokenType)

    def verify_(self) -> None:
        ndyn = len(self.dynamicSizes)
        assert isinstance(res_type := self.result.type, memref.MemRefType)
        ndyn_type = len([i for i in res_type.get_shape() if i == -1])
        if ndyn != ndyn_type:
            raise VerifyException(
                f"Expected {ndyn_type} dynamic sizes, got {ndyn}. All "
                "dynamic sizes need to be set in the alloc operation."
            )

    def __init__(
        self,
        return_type: memref.MemRefType[_Element],
        dynamic_sizes: Sequence[SSAValue | Operation] | None = None,
        host_shared: bool = False,
        async_dependencies: Sequence[SSAValue | Operation] | None = None,
        is_async: bool = False,
    ):
        token_return = [AsyncTokenType()] if is_async else []
        dynamic_sizes_vals: list[SSAValue] = (
            [SSAValue.get(e) for e in dynamic_sizes] if dynamic_sizes else []
        )
        async_dependencies_vals: list[SSAValue] = (
            [SSAValue.get(e) for e in async_dependencies] if async_dependencies else []
        )
        attributes: dict[str, Attribute] = (
            {"hostShared": UnitAttr()} if host_shared else {}
        )
        super().__init__(
            operands=[async_dependencies_vals, dynamic_sizes_vals, []],
            result_types=[return_type, token_return],
            attributes=attributes,
        )


@irdl_op_definition
class AllReduceOp(IRDLOperation):
    name = "gpu.all_reduce"
    op: AllReduceOpAttr | None = opt_prop_def(AllReduceOpAttr)
    uniform: UnitAttr | None = opt_prop_def(UnitAttr)
    operand: Operand = operand_def(Attribute)
    result: OpResult = result_def(Attribute)
    body: Region = region_def()

    traits = frozenset([IsolatedFromAbove()])

    @staticmethod
    def from_op(
        op: AllReduceOpAttr,
        operand: SSAValue | Operation,
        uniform: UnitAttr | None = None,
    ):
        return AllReduceOp.build(
            operands=[operand],
            result_types=[SSAValue.get(operand).type],
            properties={
                "op": op,
                "uniform": uniform,
            },
            regions=[Region()],
        )

    @staticmethod
    def from_body(
        body: Region, operand: SSAValue | Operation, uniform: UnitAttr | None = None
    ):
        return AllReduceOp.build(
            operands=[operand],
            result_types=[SSAValue.get(operand).type],
            properties={"uniform": uniform} if uniform is not None else {},
            regions=[body],
        )

    def verify_(self) -> None:
        if self.result.type != self.operand.type:
            raise VerifyException(
                f"Type mismatch: result type is {self.result.type}, operand type is "
                f"{self.operand.type}. They must be the same type for gpu.all_reduce"
            )

        non_empty_body = any(b.ops for b in self.body.blocks)
        op_attr = self.op is not None
        if non_empty_body == op_attr:
            if op_attr:
                raise VerifyException(
                    "gpu.all_reduce can't have both a non-empty region and an op "
                    "attribute."
                )
            else:
                raise VerifyException(
                    "gpu.all_reduce need either a non empty body or an op attribute."
                )
        if non_empty_body:
            region_args = self.body.blocks[0].args
            args_types = [r.type for r in region_args]
            if args_types != [self.result.type, self.operand.type]:
                raise VerifyException(
                    f"Expected {[str(t) for t in [self.result.type, self.operand.type]]}, "
                    f"got {[str(t) for t in args_types]}. A gpu.all_reduce's body must "
                    "have two arguments matching the result type."
                )


@irdl_op_definition
class BarrierOp(IRDLOperation):
    name = "gpu.barrier"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class BlockDimOp(IRDLOperation):
    name = "gpu.block_dim"
    dimension: DimensionAttr = prop_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class BlockIdOp(IRDLOperation):
    name = "gpu.block_id"
    dimension: DimensionAttr = prop_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class DeallocOp(IRDLOperation):
    name = "gpu.dealloc"

    asyncDependencies: VarOperand = var_operand_def(AsyncTokenType)
    buffer: Operand = operand_def(memref.MemRefType)

    irdl_options = [AttrSizedOperandSegments()]

    asyncToken: OptOpResult = opt_result_def(AsyncTokenType)

    def __init__(
        self,
        buffer: SSAValue | Operation,
        async_dependencies: Sequence[SSAValue | Operation] | None = None,
        is_async: bool = False,
    ):
        super().__init__(
            operands=[async_dependencies, buffer],
            result_types=[[AsyncTokenType()] if is_async else []],
        )


@irdl_op_definition
class MemcpyOp(IRDLOperation):
    name = "gpu.memcpy"

    asyncDependencies: VarOperand = var_operand_def(AsyncTokenType)
    dst: Operand = operand_def(memref.MemRefType)
    src: Operand = operand_def(memref.MemRefType)

    irdl_options = [AttrSizedOperandSegments()]

    asyncToken: OptOpResult = opt_result_def(AsyncTokenType)

    def __init__(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        async_dependencies: Sequence[SSAValue | Operation] | None = None,
        is_async: bool = False,
    ):
        super().__init__(
            operands=[async_dependencies, destination, source],
            result_types=[[AsyncTokenType()] if is_async else []],
        )

    def verify_(self) -> None:
        if self.src.type != self.dst.type:
            raise VerifyException(
                f"Expected {self.dst.type}, got {self.src.type}. gpu.memcpy source and "
                "destination types must match."
            )


@irdl_op_definition
class ModuleEndOp(IRDLOperation):
    name = "gpu.module_end"

    traits = traits_def(lambda: frozenset([IsTerminator(), HasParent(ModuleOp)]))

    def __init__(self):
        super().__init__()


@irdl_op_definition
class ModuleOp(IRDLOperation):
    name = "gpu.module"

    body: Region = region_def("single_block")
    sym_name: StringAttr = attr_def(StringAttr)

    traits = frozenset(
        [
            IsolatedFromAbove(),
            SingleBlockImplicitTerminator(ModuleEndOp),
            SymbolOpInterface(),
            SymbolTable(),
        ]
    )

    def __init__(self, name: SymbolRefAttr, ops: Sequence[Operation]):
        super().__init__(attributes={"sym_name": name}, regions=[ops])


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "gpu.func"

    body: Region = region_def()
    sym_name: StringAttr = attr_def(StringAttr)
    function_type: FunctionType = prop_def(FunctionType)
    kernel: UnitAttr | None = opt_prop_def(UnitAttr)
    known_block_size: DenseArrayBase | None = opt_attr_def(
        DenseArrayBase, attr_name="gpu.known_block_size"
    )
    known_grid_size: DenseArrayBase | None = opt_attr_def(
        DenseArrayBase, attr_name="gpu.known_grid_size"
    )

    traits = frozenset([IsolatedFromAbove(), HasParent(ModuleOp), SymbolOpInterface()])

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        kernel: bool | None = None,
        knwown_block_size: Sequence[int] | None = None,
        knwown_grid_size: Sequence[int] | None = None,
    ):
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        attributes: dict[str, Attribute | None] = {"sym_name": StringAttr(name)}
        properties: dict[str, Attribute | None] = {
            "function_type": function_type,
        }
        if knwown_block_size is not None:
            attributes["gpu.known_block_size"] = ArrayAttr(
                IntegerAttr(i, i32) for i in knwown_block_size
            )
        if knwown_grid_size is not None:
            attributes["gpu.known_grid_size"] = ArrayAttr(
                IntegerAttr(i, i32) for i in knwown_grid_size
            )
        if kernel:
            properties["kernel"] = UnitAttr()
        super().__init__(properties=properties, attributes=attributes, regions=[region])

    def verify_(self):
        entry_block: Block = self.body.blocks[0]
        function_inputs = self.function_type.inputs.data
        block_arg_types = tuple(a.type for a in entry_block.args)
        if function_inputs != block_arg_types:
            raise VerifyException(
                "Expected first entry block arguments to have the same types as the "
                "function input types"
            )
        if (self.kernel is not None) and (len(self.function_type.outputs) != 0):
            raise VerifyException("Expected void return type for kernel function")


@irdl_op_definition
class GlobalIdOp(IRDLOperation):
    name = "gpu.global_id"
    dimension: DimensionAttr = prop_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class GridDimOp(IRDLOperation):
    name = "gpu.grid_dim"
    dimension: DimensionAttr = prop_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class HostRegisterOp(IRDLOperation):
    """
    This op maps the provided host buffer into the device address space.

    This operation may not be supported in every environment, there is not yet a way to
    check at runtime whether this feature is supported.
    Writes from the host are guaranteed to be visible to device kernels that are launched
    afterwards. Writes from the device are guaranteed to be visible on the host after
    synchronizing with the device kernel completion.
    """

    name = "gpu.host_register"

    value: Operand = operand_def(memref.UnrankedMemrefType)

    def __init__(self, memref: SSAValue | Operation):
        super().__init__(operands=[SSAValue.get(memref)])


@irdl_op_definition
class HostUnregisterOp(IRDLOperation):
    """
    Unregisters a memref for access from device.
    """

    name = "gpu.host_unregister"

    value: Operand = operand_def(memref.UnrankedMemrefType)

    def __init__(self, memref: SSAValue | Operation):
        super().__init__(operands=[SSAValue.get(memref)])


@irdl_op_definition
class LaneIdOp(IRDLOperation):
    name = "gpu.lane_id"
    result: OpResult = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


@irdl_op_definition
class LaunchOp(IRDLOperation):
    name = "gpu.launch"
    asyncDependencies: VarOperand = var_operand_def(AsyncTokenType)
    gridSizeX: Operand = operand_def(IndexType)
    gridSizeY: Operand = operand_def(IndexType)
    gridSizeZ: Operand = operand_def(IndexType)
    blockSizeX: Operand = operand_def(IndexType)
    blockSizeY: Operand = operand_def(IndexType)
    blockSizeZ: Operand = operand_def(IndexType)
    dynamicSharedMemorySize: OptOperand = opt_operand_def(i32)
    asyncToken: OptOpResult = opt_result_def(AsyncTokenType)
    body: Region = region_def()
    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        body: Region,
        gridSize: Sequence[SSAValue | Operation],
        blockSize: Sequence[SSAValue | Operation],
        async_launch: bool = False,
        asyncDependencies: Sequence[SSAValue | Operation] | None = None,
        dynamicSharedMemorySize: SSAValue | Operation | None = None,
    ):
        if len(gridSize) != 3:
            raise ValueError(f"LaunchOp must have 3 gridSizes, got {len(gridSize)}")
        if len(blockSize) != 3:
            raise ValueError(f"LaunchOp must have 3 blockSizes, got {len(blockSize)}")
        operands = [
            []
            if asyncDependencies is None
            else [SSAValue.get(a) for a in asyncDependencies]
        ]

        operands += [SSAValue.get(gs) for gs in gridSize]
        operands += [SSAValue.get(bs) for bs in blockSize]
        operands += [
            []
            if dynamicSharedMemorySize is None
            else [SSAValue.get(dynamicSharedMemorySize)]
        ]
        super().__init__(
            operands=operands,
            result_types=[[AsyncTokenType()] if async_launch else []],
            regions=[body],
        )

    def verify_(self) -> None:
        if not any(b.ops for b in self.body.blocks):
            raise VerifyException("gpu.launch requires a non-empty body.")
        body_args = self.body.blocks[0].args
        args_type = [a.type for a in body_args]
        if args_type != [IndexType()] * 12:
            raise VerifyException(
                f"Expected [12 x {str(IndexType())}], got {[str(t) for t in args_type]}. "
                "gpu.launch's body arguments are 12 index arguments, with 3 block "
                "indices, 3 block sizes, 3 thread indices, and 3 thread counts"
            )


@irdl_op_definition
class LaunchFuncOp(IRDLOperation):
    """
    Launch a kernel function on the specified grid of thread blocks. gpu.launch
    operations are lowered to gpu.launch_func operations by outlining the kernel body
    into a function in a dedicated module, which reflects the separate compilation
    process. The kernel function is required to have the gpu.kernel attribute. The
    module containing the kernel function is required to be a gpu.module. And finally,
    the module containing the kernel module (which thus cannot be the top-level module)
    is required to have the gpu.container_module attribute. The gpu.launch_func operation
    has a symbol attribute named kernel to identify the fully specified kernel function
    to launch (both the gpu.module and func).

    The gpu.launch_func supports async dependencies: the kernel does not start executing
    until the ops producing those async dependencies have completed.

    By default, the host implicitly blocks until kernel execution has completed. If
    the async keyword is present, the host does not block but instead a !gpu.async.token
    is returned. Other async GPU ops can take this token as dependency.

    The operation requires at least the grid and block sizes along the x,y,z dimensions
    as arguments. When a lower-dimensional kernel is required, unused sizes must be
    explicitly set to 1.

    The remaining operands are optional. The first optional operand corresponds to the
    amount of dynamic shared memory a kernel's workgroup should be allocated; when this
    operand is not present, a zero size is assumed.

    The remaining operands if present are passed as arguments to the kernel function.
    """

    name = "gpu.launch_func"
    asyncDependencies: VarOperand = var_operand_def(AsyncTokenType)
    gridSizeX: Operand = operand_def(IndexType)
    gridSizeY: Operand = operand_def(IndexType)
    gridSizeZ: Operand = operand_def(IndexType)
    blockSizeX: Operand = operand_def(IndexType)
    blockSizeY: Operand = operand_def(IndexType)
    blockSizeZ: Operand = operand_def(IndexType)
    clusterSizeX: OptOperand = opt_operand_def(IndexType)
    clusterSizeY: OptOperand = opt_operand_def(IndexType)
    clusterSizeZ: OptOperand = opt_operand_def(IndexType)
    dynamicSharedMemorySize: OptOperand = opt_operand_def(i32)
    kernelOperands: VarOperand = var_operand_def()
    asyncObject: OptOperand = opt_operand_def()

    asyncToken: OptOpResult = opt_result_def(AsyncTokenType)

    kernel: SymbolRefAttr = prop_def(SymbolRefAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        func: SymbolRefAttr,
        gridSize: Sequence[SSAValue | Operation],
        blockSize: Sequence[SSAValue | Operation],
        clusterSize: Sequence[SSAValue | Operation] | None = None,
        kernelOperands: Sequence[SSAValue | Operation] | None = None,
        async_launch: bool = False,
        asyncDependencies: Sequence[SSAValue | Operation] | None = None,
        dynamicSharedMemorySize: SSAValue | Operation | None = None,
    ):
        if len(gridSize) != 3:
            raise ValueError(f"LaunchOp must have 3 gridSizes, got {len(gridSize)}")
        if len(blockSize) != 3:
            raise ValueError(f"LaunchOp must have 3 blockSizes, got {len(blockSize)}")
        clusterSizeOperands: Sequence[
            SSAValue | Operation | Sequence[SSAValue | Operation]
        ]
        if clusterSize is None:
            clusterSizeOperands = [[], [], []]
        else:
            clusterSizeOperands = clusterSize
        if len(clusterSizeOperands) != 3:
            raise ValueError(
                f"LaunchFuncOp must have 3 cluterSizes if any, got {len(clusterSizeOperands)}"
            )

        super().__init__(
            operands=[
                asyncDependencies,
                *gridSize,
                *blockSize,
                *clusterSizeOperands,
                dynamicSharedMemorySize,
                kernelOperands,
                [],
            ],
            result_types=[[AsyncTokenType()] if async_launch else []],
            properties={"kernel": func},
        )


@irdl_op_definition
class NumSubgroupsOp(IRDLOperation):
    name = "gpu.num_subgroups"
    result: OpResult = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "gpu.return"

    args: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator(), HasParent(FuncOp)])

    def __init__(self, operands: Sequence[SSAValue | Operation]):
        super().__init__(operands=[operands])


@irdl_op_definition
class SetDefaultDeviceOp(IRDLOperation):
    name = "gpu.set_default_device"
    devIndex: Operand = operand_def(i32)

    def __init__(self, devIndex: SSAValue | Operation):
        super().__init__(operands=[SSAValue.get(devIndex)])


@irdl_op_definition
class SubgroupIdOp(IRDLOperation):
    name = "gpu.subgroup_id"
    result: OpResult = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


@irdl_op_definition
class SubgroupSizeOp(IRDLOperation):
    name = "gpu.subgroup_size"
    result: OpResult = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


@irdl_op_definition
class TerminatorOp(IRDLOperation):
    name = "gpu.terminator"

    traits = frozenset([HasParent(LaunchOp), IsTerminator()])

    def __init__(self):
        super().__init__()


@irdl_op_definition
class ThreadIdOp(IRDLOperation):
    name = "gpu.thread_id"
    dimension: DimensionAttr = prop_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "gpu.yield"
    values: VarOperand = var_operand_def(Attribute)

    def __init__(self, operands: Sequence[SSAValue | Operation]):
        super().__init__(operands=[operands])

    traits = frozenset([IsTerminator()])

    def verify_(self) -> None:
        op = self.parent_op()
        if op is not None:
            yield_type = [o.type for o in self.values]
            result_type = [r.type for r in op.results]
            if yield_type != result_type:
                raise VerifyException(
                    f"Expected {[str(t) for t in result_type]}, got {[str(t) for t in yield_type]}. The gpu.yield values "
                    "types must match its enclosing operation result types."
                )


GPU = Dialect(
    "gpu",
    [
        AllocOp,
        AllReduceOp,
        BarrierOp,
        BlockDimOp,
        BlockIdOp,
        DeallocOp,
        FuncOp,
        GlobalIdOp,
        GridDimOp,
        HostRegisterOp,
        HostUnregisterOp,
        LaneIdOp,
        LaunchOp,
        LaunchFuncOp,
        MemcpyOp,
        ModuleOp,
        ModuleEndOp,
        NumSubgroupsOp,
        ReturnOp,
        SetDefaultDeviceOp,
        SubgroupIdOp,
        SubgroupSizeOp,
        TerminatorOp,
        ThreadIdOp,
        YieldOp,
    ],
    [
        AllReduceOpAttr,
        DimensionAttr,
        ProcessorAttr,
        LoopDimMapAttr,
    ],
)
