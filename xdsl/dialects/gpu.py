from __future__ import annotations

from typing import Generic, Sequence, TypeVar

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    i32,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.ir.core import Block
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
    opt_result_def,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class AsyncTokenType(ParametrizedAttribute, TypeAttribute):
    name = "gpu.async.token"


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


T = TypeVar("T", bound=_AllReduceOperationAttr | _DimensionAttr, covariant=True)


@irdl_attr_definition
class _GPUAttr(ParametrizedAttribute, Generic[T]):
    name = "gpu"

    value: ParameterDef[T]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters(
            "<",
            ": gpu attributes currently have the #gpu<name value> syntax.",
        )
        if parser.parse_optional_keyword("dim"):
            attrtype = _DimensionAttr
            vtok = parser.parse_optional_identifier()
            if vtok not in ["x", "y", "z"]:
                parser.raise_error(
                    f"Unexpected dim {vtok}. A gpu dim can only be x, y, or z",
                )

        elif parser.parse_optional_keyword("all_reduce_op"):
            attrtype = _AllReduceOperationAttr
            vtok = parser.parse_optional_identifier()
            if vtok not in ["add", "and", "max", "min", "mul", "or", "xor"]:
                parser.raise_error(
                    f"Unexpected op {vtok}. A gpu all_reduce_op can only be add, "
                    "and, max, min, mul, or, or xor ",
                )
        else:
            parser.raise_error(f"'dim' or 'all_reduce_op' expected")
        parser.parse_characters(
            ">",
            f". gpu attributes currently have the #gpu<name value> syntax.",
        )
        return [attrtype([StringAttr(vtok)])]

    @staticmethod
    def from_op(value: str) -> AllReduceOperationAttr:
        return AllReduceOperationAttr([_AllReduceOperationAttr([StringAttr(value)])])

    @property
    def data(self) -> str:
        return self.value.param.data

    @staticmethod
    def from_dimension(value: str) -> DimensionAttr:
        return DimensionAttr([_DimensionAttr([StringAttr(value)])])

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        self.value.print_parameters(printer)
        printer.print_string(">")


DimensionAttr = _GPUAttr[_DimensionAttr]
AllReduceOperationAttr = _GPUAttr[_AllReduceOperationAttr]

_Element = TypeVar("_Element", bound=Attribute, covariant=True)


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "gpu.alloc"
    hostShared: UnitAttr | None = opt_attr_def(UnitAttr)
    asyncDependencies: VarOperand = var_operand_def(AsyncTokenType)
    dynamicSizes: VarOperand = var_operand_def(IndexType)
    symbolOperands: VarOperand = var_operand_def(IndexType)

    irdl_options = [AttrSizedOperandSegments()]

    result: OpResult = result_def(memref.MemRefType[Attribute])
    asyncToken: OptOpResult = opt_result_def(AsyncTokenType)

    def verify_(self) -> None:
        ndyn = len(self.dynamicSizes)
        assert isinstance(self.result.type, memref.MemRefType)
        res_type: memref.MemRefType[Attribute] = self.result.type
        ndyn_type = len([i for i in res_type.shape.data if i.value.data == -1])
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
        return super().__init__(
            operands=[async_dependencies_vals, dynamic_sizes_vals, []],
            result_types=[return_type, token_return],
            attributes=attributes,
        )


@irdl_op_definition
class AllReduceOp(IRDLOperation):
    name = "gpu.all_reduce"
    op: AllReduceOperationAttr | None = opt_attr_def(AllReduceOperationAttr)
    uniform: UnitAttr | None = opt_attr_def(UnitAttr)
    operand: Operand = operand_def(Attribute)
    result: OpResult = result_def(Attribute)
    body: Region = region_def()

    traits = frozenset([IsolatedFromAbove()])

    @staticmethod
    def from_op(
        op: AllReduceOperationAttr,
        operand: SSAValue | Operation,
        uniform: UnitAttr | None = None,
    ):
        return AllReduceOp.build(
            operands=[operand],
            result_types=[SSAValue.get(operand).type],
            attributes={
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
            attributes={"uniform": uniform} if uniform is not None else {},
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
                    f"gpu.all_reduce can't have both a non-empty region and an op "
                    "attribute."
                )
            else:
                raise VerifyException(
                    f"gpu.all_reduce need either a non empty body or an op attribute."
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
        return super().__init__()


@irdl_op_definition
class BlockDimOp(IRDLOperation):
    name = "gpu.block_dim"
    dimension: DimensionAttr = attr_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        return super().__init__(
            result_types=[IndexType()], attributes={"dimension": dim}
        )


@irdl_op_definition
class BlockIdOp(IRDLOperation):
    name = "gpu.block_id"
    dimension: DimensionAttr = attr_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        return super().__init__(
            result_types=[IndexType()], attributes={"dimension": dim}
        )


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
        return super().__init__(
            operands=[async_dependencies, buffer],
            result_types=[[AsyncTokenType()] if is_async else []],
        )


@irdl_op_definition
class MemcpyOp(IRDLOperation):
    name = "gpu.memcpy"

    asyncDependencies: VarOperand = var_operand_def(AsyncTokenType)
    src: Operand = operand_def(memref.MemRefType)
    dst: Operand = operand_def(memref.MemRefType)

    irdl_options = [AttrSizedOperandSegments()]

    asyncToken: OptOpResult = opt_result_def(AsyncTokenType)

    def __init__(
        self,
        source: SSAValue | Operation,
        destination: SSAValue | Operation,
        async_dependencies: Sequence[SSAValue | Operation] | None = None,
        is_async: bool = False,
    ):
        return super().__init__(
            operands=[async_dependencies, source, destination],
            result_types=[[AsyncTokenType()] if is_async else []],
        )

    def verify_(self) -> None:
        if self.src.type != self.dst.type:
            raise VerifyException(
                f"Expected {self.src.type}, got {self.dst.type}. gpu.memcpy source and "
                "destination types must match."
            )


@irdl_op_definition
class ModuleEndOp(IRDLOperation):
    name = "gpu.module_end"

    # TODO circular dependency disallows this set of traits
    # tracked by gh issues https://github.com/xdslproject/xdsl/issues/1218
    # traits = frozenset([HasParent(ModuleOp), IsTerminator()])
    traits = frozenset([IsTerminator()])

    def __init__(self):
        return super().__init__()


@irdl_op_definition
class ModuleOp(IRDLOperation):
    name = "gpu.module"

    body: Region = region_def("single_block")
    sym_name: StringAttr = attr_def(StringAttr)

    traits = frozenset(
        [IsolatedFromAbove(), SingleBlockImplicitTerminator(ModuleEndOp)]
    )

    def __init__(self, name: SymbolRefAttr, ops: Sequence[Operation]):
        super().__init__(attributes={"sym_name": name}, regions=[ops])


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "gpu.func"

    body: Region = region_def()
    sym_name: StringAttr = attr_def(StringAttr)
    function_type: FunctionType = attr_def(FunctionType)
    kernel: UnitAttr | None = opt_attr_def(UnitAttr)

    traits = frozenset([IsolatedFromAbove(), HasParent(ModuleOp)])

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
            raise VerifyException(f"Expected void return type for kernel function")


@irdl_op_definition
class GlobalIdOp(IRDLOperation):
    name = "gpu.global_id"
    dimension: DimensionAttr = attr_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        return super().__init__(
            result_types=[IndexType()], attributes={"dimension": dim}
        )


@irdl_op_definition
class GridDimOp(IRDLOperation):
    name = "gpu.grid_dim"
    dimension: DimensionAttr = attr_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        return super().__init__(
            result_types=[IndexType()], attributes={"dimension": dim}
        )


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
        return super().__init__(operands=[SSAValue.get(memref)])


@irdl_op_definition
class HostUnregisterOp(IRDLOperation):
    """
    Unregisters a memref for access from device.
    """

    name = "gpu.host_unregister"

    value: Operand = operand_def(memref.UnrankedMemrefType)

    def __init__(self, memref: SSAValue | Operation):
        return super().__init__(operands=[SSAValue.get(memref)])


@irdl_op_definition
class LaneIdOp(IRDLOperation):
    name = "gpu.lane_id"
    result: OpResult = result_def(IndexType)

    def __init__(self):
        return super().__init__(result_types=[IndexType()])


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
    irdl_options = [AttrSizedOperandSegments()]

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
        return super().__init__(
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
    dynamicSharedMemorySize: OptOperand = opt_operand_def(i32)
    kernelOperands: VarOperand = var_operand_def()

    asyncToken: OptOpResult = opt_result_def(AsyncTokenType)

    kernel: SymbolRefAttr = attr_def(SymbolRefAttr)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        func: SymbolRefAttr,
        gridSize: Sequence[SSAValue | Operation],
        blockSize: Sequence[SSAValue | Operation],
        kernelOperands: Sequence[SSAValue | Operation] | None = None,
        async_launch: bool = False,
        asyncDependencies: Sequence[SSAValue | Operation] | None = None,
        dynamicSharedMemorySize: SSAValue | Operation | None = None,
    ):
        if len(gridSize) != 3:
            raise ValueError(f"LaunchOp must have 3 gridSizes, got {len(gridSize)}")
        if len(blockSize) != 3:
            raise ValueError(f"LaunchOp must have 3 blockSizes, got {len(blockSize)}")

        return super().__init__(
            operands=[
                asyncDependencies,
                *gridSize,
                *blockSize,
                dynamicSharedMemorySize,
                kernelOperands,
            ],
            result_types=[[AsyncTokenType()] if async_launch else []],
            attributes={"kernel": func},
        )


@irdl_op_definition
class NumSubgroupsOp(IRDLOperation):
    name = "gpu.num_subgroups"
    result: OpResult = result_def(IndexType)

    def __init__(self):
        return super().__init__(result_types=[IndexType()])


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "gpu.return"

    args: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator(), HasParent(FuncOp)])

    def __init__(self, operands: Sequence[SSAValue | Operation]):
        return super().__init__([operands])


@irdl_op_definition
class SetDefaultDeviceOp(IRDLOperation):
    name = "gpu.set_default_device"
    devIndex: Operand = operand_def(i32)

    def __init__(self, devIndex: SSAValue | Operation):
        return super().__init__(operands=[SSAValue.get(devIndex)])


@irdl_op_definition
class SubgroupIdOp(IRDLOperation):
    name = "gpu.subgroup_id"
    result: OpResult = result_def(IndexType)

    def __init__(self):
        return super().__init__(result_types=[IndexType()])


@irdl_op_definition
class SubgroupSizeOp(IRDLOperation):
    name = "gpu.subgroup_size"
    result: OpResult = result_def(IndexType)

    def __init__(self):
        return super().__init__(result_types=[IndexType()])


@irdl_op_definition
class TerminatorOp(IRDLOperation):
    name = "gpu.terminator"

    traits = frozenset([HasParent(LaunchOp), IsTerminator()])

    def __init__(self):
        return super().__init__()


@irdl_op_definition
class ThreadIdOp(IRDLOperation):
    name = "gpu.thread_id"
    dimension: DimensionAttr = attr_def(DimensionAttr)
    result: OpResult = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        return super().__init__(
            result_types=[IndexType()], attributes={"dimension": dim}
        )


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "gpu.yield"
    values: VarOperand = var_operand_def(Attribute)

    def __init__(self, operands: Sequence[SSAValue | Operation]):
        return super().__init__([operands])

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


# _GPUAttr has to be registered instead of DimensionAttr and AllReduceOperationAttr here.
# This is a hack to fit MLIR's syntax in xDSL's way of parsing attributes, without making GPU builtin.
# Hopefully MLIR will parse it in a more xDSL-friendly way soon, so all that can be factored in proper xDSL
# atrributes.
GPU = Dialect(
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
    [_GPUAttr],
)
