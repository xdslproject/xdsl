from abc import ABC
from collections.abc import Sequence
from enum import auto
from typing import ClassVar, TypeAlias

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    I16,
    I64,
    ArrayAttr,
    BytesAttr,
    DenseArrayBase,
    FlatSymbolRefAttrConstr,
    IndexType,
    IntAttr,
    IntegerAttr,
    MemRefType,
    ShapedType,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    TensorType,
    UnitAttr,
    i16,
    i64,
)
from xdsl.dialects.utils import DimensionList, DynamicIndexList, EnumAttribute
from xdsl.ir import (
    Attribute,
    Dialect,
    OpaqueSyntaxAttribute,
    Operation,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    AtLeast,
    AttrSizedOperandSegments,
    IRDLOperation,
    RangeOf,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    Pure,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.str_enum import StrEnum
from xdsl.utils.type import get_element_type_or_self

################################################################################
# Types and attributes                                                         #
################################################################################

ShardAxis: TypeAlias = I16
"""
The type used to represent numbers on a grid axis.

See [the MLIR definition](https://github.com/llvm/llvm-project/blob/fef02d48c08db859ef83f84232ed78bd9d1c323a/mlir/include/mlir/Dialect/Shard/IR/ShardBase.td#L39).
"""


ShardAxesAttr: TypeAlias = DenseArrayBase[ShardAxis]
"""
The type used to represent a list of grid axes.

See [the MLIR definition](https://github.com/llvm/llvm-project/blob/fef02d48c08db859ef83f84232ed78bd9d1c323a/mlir/include/mlir/Dialect/Shard/IR/ShardBase.td#L40).
"""


def _parse_grid_axes_attr(parser: AttrParser) -> ShardAxesAttr:
    """
    Parses a single ShardAxesAttr, e.g. [1, 4, 7, 8]
    """
    elements = parser.parse_comma_separated_list(
        parser.Delimiter.SQUARE,
        parser.parse_integer,
    )

    return ShardAxesAttr(i16, BytesAttr(i16.pack(elements)))


def _print_sublist(printer: Printer, sublist: ShardAxesAttr) -> None:
    """
    Prints a single ShardAxesAttr, e.g. [1, 4, 6, 8]
    """
    with printer.in_square_brackets():
        printer.print_list(sublist.get_values(), printer.print_int)


@irdl_attr_definition
class ShardAxesArrayAttr(ParametrizedAttribute, OpaqueSyntaxAttribute):
    """
    ShardAxesArrayAttr attribute for representing multiple grid axes.

    Reflects [the MLIR attribute](https://github.com/llvm/llvm-project/blob/fef02d48c08db859ef83f84232ed78bd9d1c323a/mlir/include/mlir/Dialect/Shard/IR/ShardBase.td#L83).
    """

    name = "shard.axisarray"

    axes: ArrayAttr[ShardAxesAttr]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """
        Parses a ShardAxesArrayAttr, which has the syntax of a list
        of lists, e.g.:

        [[1, 2, 3], [], [4, 5]]
        """
        axes = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE,
            lambda: _parse_grid_axes_attr(parser),
        )

        return (ArrayAttr(axes),)

    def print_parameters(self, printer: Printer) -> None:
        """
        Prints a ShardAxesArrayAttr, which has the syntax of a list
        of lists, e.g.:

        [[1, 2, 3], [], [4, 5]]
        """
        with printer.in_square_brackets():
            printer.print_list(
                self.axes.data,
                lambda x: _print_sublist(printer, x),
            )


class ReductionKind(StrEnum):
    "Reduction kind for grid dialect"

    SUM = auto()
    MAX = auto()
    MIN = auto()
    PRODUCT = auto()
    AVERAGE = auto()
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    GENERIC = auto()


@irdl_attr_definition
class ReductionKindAttr(EnumAttribute[ReductionKind], SpacedOpaqueSyntaxAttribute):
    name = "shard.partial"

    assembly_format = "$value"


@irdl_attr_definition
class ShardingType(ParametrizedAttribute, TypeAttribute):
    name = "shard.sharding"


NonZeroRankedTensorConstr = TensorType.constr(
    shape=ArrayAttr.constr(RangeOf.get(IntAttr).of_length(AtLeast(1)))
)
"""
Constrains a ranked tensor to have a non-zero rank.

Reflects MLIR's `AnyNon0RankedTensor`.
"""


################################################################################
# Grid symbol verification                                                     #
################################################################################


class GridSymbolUserOperation(IRDLOperation, ABC):
    """
    Base class for ops referencing a `shard.grid` via their `grid` property.
    """

    grid = prop_def(FlatSymbolRefAttrConstr)

    def get_grid(self) -> "GridOp":
        """
        Looks up the `shard.grid` referenced by `grid`, raising a
        `VerifyException` if it is not found.
        """
        found = SymbolTable.lookup_symbol(self, self.grid)
        if not isinstance(found, GridOp):
            raise VerifyException(
                f'Undefined required grid symbol "{self.grid.root_reference.data}".'
            )
        return found

    def verify_(self) -> None:
        self.get_grid()


def _verify_grid_axes(axes: Sequence[int], grid: "GridOp") -> None:
    """
    Verifies that `axes` are unique and in bounds of `grid`.
    """
    if len(set(axes)) != len(axes):
        raise VerifyException("Grid axes contains duplicate elements.")

    rank = len(grid.shape)
    for axis in axes:
        if not 0 <= axis < rank:
            raise VerifyException(
                f"0-based grid axis index {axis} is out of bounds. The referenced "
                f'grid "{grid.sym_name.data}" is of rank {rank}.'
            )


################################################################################
# Collective shape verification                                                #
################################################################################


def _dim_mul(lhs: int, rhs: int) -> int:
    if DYNAMIC_INDEX in (lhs, rhs):
        return DYNAMIC_INDEX
    return lhs * rhs


def _dim_div(lhs: int, rhs: int) -> int:
    if DYNAMIC_INDEX in (lhs, rhs):
        return DYNAMIC_INDEX
    return lhs // rhs


def _verify_dimension_compatibility(expected: int, actual: int, axis: int) -> None:
    if actual != DYNAMIC_INDEX and expected != actual:
        expected_str = "dynamic" if expected == DYNAMIC_INDEX else str(expected)
        raise VerifyException(
            f"Dimension size mismatch for result axis {axis}. "
            f"Expected {expected_str}, but got {actual}."
        )


def _verify_same_op_res_elem_t(op: Operation):
    if len(op.results) < 1 or len(op.operands) < 1:
        raise VerifyException(f"'{op.name}' requires at least one result or operand")

    element_type = get_element_type_or_self(op.result_types[0])
    if any(
        get_element_type_or_self(t) != element_type
        for t in (*op.result_types[1:], *op.operand_types)
    ):
        raise VerifyException(
            f"'{op.name}' op requires the same element type for all operands "
            "and results"
        )


def _verify_same_op_res_rank(op: Operation) -> None:
    if not op.operands:
        raise VerifyException(f"'{op.name}' requires at least one operand")

    operand_ranks = [
        t.get_num_dims() for t in op.operand_types if isinstance(t, ShapedType)
    ]
    result_ranks = [
        t.get_num_dims() for t in op.result_types if isinstance(t, ShapedType)
    ]
    ranks = operand_ranks or result_ranks
    if not ranks:
        return

    if any(r != ranks[0] for r in operand_ranks):
        raise VerifyException(f"'{op.name}' op operands don't have matching ranks")
    if any(r != ranks[0] for r in result_ranks):
        raise VerifyException(
            f"'{op.name}' op result type has different rank than operands"
        )


def _verify_scatter_or_slice_operand_and_result_shape(
    operand: SSAValue[ShapedType],
    result: SSAValue[ShapedType],
    tensor_axis: int,
    group_size: int,
) -> None:
    operand_shape, result_shape = operand.type.get_shape(), result.type.get_shape()
    rank = len(operand_shape)
    if not 0 <= tensor_axis < rank:
        raise VerifyException(
            f"Tensor axis {tensor_axis} is out of bounds [0, {rank})."
        )

    for axis, (operand_dim, result_dim) in enumerate(zip(operand_shape, result_shape)):
        if axis != tensor_axis:
            _verify_dimension_compatibility(operand_dim, result_dim, axis)

    operand_dim = operand_shape[tensor_axis]
    if DYNAMIC_INDEX not in (operand_dim, group_size) and operand_dim % group_size:
        raise VerifyException(
            f"Operand dimension size {operand_dim} is not divisible by collective "
            f"device group size {group_size} for tensor axis {tensor_axis}."
        )
    _verify_dimension_compatibility(
        _dim_div(operand_dim, group_size), result_shape[tensor_axis], tensor_axis
    )


################################################################################
# Collective communication ops                                                 #
################################################################################


class CollectiveCommunicationOperation(GridSymbolUserOperation, ABC):
    """
    Base class for collective communication ops.
    """

    grid_axes = prop_def(
        ShardAxesAttr, default_value=ShardAxesAttr(i16, BytesAttr(b""))
    )

    def verify_(self) -> None:
        super().verify_()
        _verify_grid_axes(self.grid_axes.get_values(), self.get_grid())

    def collective_process_group_size(self, grid: "GridOp") -> int:
        """
        Number of devices in each collective group, or `DYNAMIC_INDEX` if unknown.
        """
        grid_shape = grid.shape.get_values()
        size = 1
        for axis in self.grid_axes.get_values():
            size = _dim_mul(size, grid_shape[axis])
        return size


@irdl_op_definition
class AllGatherOp(CollectiveCommunicationOperation):
    """
    All-gather over a device grid.

    Concatenates all tensor slices from a device group defined by `grid_axes`
    along the tensor dimension `gather_axis` and replicates the result across all
    devices in the group.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardall_gather-shardallgatherop).
    """

    name = "shard.all_gather"

    input = operand_def(NonZeroRankedTensorConstr)
    gather_axis = prop_def(IntegerAttr.constr(IndexType))

    result = result_def(NonZeroRankedTensorConstr)

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "`gather_axis` `=` $gather_axis "
        + "attr-dict `:` type($input) `->` type($result)"
    )

    def verify_(self) -> None:
        super().verify_()
        _verify_same_op_res_elem_t(self)
        _verify_same_op_res_rank(self)
        gather_axis = self.gather_axis.value.data
        group_size = self.collective_process_group_size(self.get_grid())
        assert isinstance(self.input.type, ShapedType)
        assert isinstance(self.result.type, ShapedType)
        operand_shape, result_shape = (
            self.input.type.get_shape(),
            self.result.type.get_shape(),
        )
        rank = len(result_shape)
        if not 0 <= gather_axis < rank:
            raise VerifyException(
                f"Gather axis {gather_axis} is out of bounds [0, {rank})."
            )

        for axis, (operand_dim, result_dim) in enumerate(
            zip(operand_shape, result_shape)
        ):
            expected = (
                _dim_mul(group_size, operand_dim)
                if axis == gather_axis
                else operand_dim
            )
            _verify_dimension_compatibility(expected, result_dim, axis)


@irdl_op_definition
class AllReduceOp(CollectiveCommunicationOperation):
    """
    All-reduce over a device grid.

    Within each device group reduce the input using the `reduction` method.
    Each device in a group receives a replicated copy of the reduction result.
    The accumulation element type is determined by the result type.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardall_reduce-shardallreduceop).
    """

    name = "shard.all_reduce"

    input = operand_def(MemRefType.constr() | TensorType.constr())
    reduction = prop_def(
        ReductionKindAttr, default_value=ReductionKindAttr(ReductionKind.SUM)
    )

    result = result_def(MemRefType.constr() | TensorType.constr())

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "(`reduction` `=` $reduction^)? "
        + "attr-dict `:` type($input) `->` type($result)"
    )


@irdl_op_definition
class AllSliceOp(CollectiveCommunicationOperation):
    """
    All-slice over a device grid.

    Within each device group defined by `grid_axes`, slices the input tensor
    along the `slice_axis` dimension. It can be viewed as the inverse of an
    all-gather if the input data is replicated along the `slice_axis`. Does not
    involve any communication between devices.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardall_slice-shardallsliceop).
    """

    name = "shard.all_slice"

    input = operand_def(NonZeroRankedTensorConstr)
    slice_axis = prop_def(IntegerAttr.constr(IndexType))

    result = result_def(NonZeroRankedTensorConstr)

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "`slice_axis` `=` $slice_axis "
        + "attr-dict `:` type($input) `->` type($result)"
    )

    def verify_(self) -> None:
        super().verify_()
        _verify_same_op_res_elem_t(self)
        _verify_same_op_res_rank(self)
        _verify_scatter_or_slice_operand_and_result_shape(
            SSAValue.get(self.input, type=ShapedType),
            self.result,
            self.slice_axis.value.data,
            self.collective_process_group_size(self.get_grid()),
        )


@irdl_op_definition
class BroadcastOp(CollectiveCommunicationOperation):
    """
    Broadcast tensor from one device to many devices.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardbroadcast-shardbroadcastop).
    """

    name = "shard.broadcast"

    input = operand_def(TensorType)
    root = prop_def(DenseArrayBase[I64])
    root_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "`root` `=` custom<DynamicIndexList>($root_dynamic, $root) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class GatherOp(CollectiveCommunicationOperation):
    """
    Gather tensor shards from many devices to a single device.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardgather-shardgatherop).
    """

    name = "shard.gather"

    input = operand_def(TensorType)
    gather_axis = prop_def(IntegerAttr.constr(IndexType))
    root = prop_def(DenseArrayBase[I64])
    root_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "`gather_axis` `=` $gather_axis "
        + "`root` `=` custom<DynamicIndexList>($root_dynamic, $root) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class ScatterOp(CollectiveCommunicationOperation):
    """
    Scatter tensor over a device shard.

    For each device group split the input tensor on the `root` device along
    axis `scatter_axis` and scatter the parts across the group devices.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardscatter-shardscatterop).
    """

    name = "shard.scatter"

    input = operand_def(TensorType)
    scatter_axis = prop_def(IntegerAttr.constr(IndexType))

    root = prop_def(DenseArrayBase[I64])
    root_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    traits = traits_def(
        Pure(),
    )

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "`scatter_axis` `=` $scatter_axis "
        + "`root` `=` custom<DynamicIndexList>($root_dynamic, $root) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class RecvOp(CollectiveCommunicationOperation):
    """
    Receive from a device within a device group.
    """

    name = "shard.recv"

    input = operand_def(TensorType)
    source = opt_prop_def(DenseArrayBase[I64])
    source_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "(`source` `=` custom<DynamicIndexList>($source_dynamic, $source)^)? "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class ReduceOp(CollectiveCommunicationOperation):
    """
    Reduce over a device grid.

    Within each device group reduce the input tensor using the `reduction`
    method. The result is returned on the `root` device of each group and is
    undefined on all other devices.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardreduce-shardreduceop).
    """

    name = "shard.reduce"

    input = operand_def(TensorType)
    reduction = prop_def(
        ReductionKindAttr, default_value=ReductionKindAttr(ReductionKind.SUM)
    )
    root = prop_def(DenseArrayBase[I64])
    root_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "(`reduction` `=` $reduction^)? "
        + "`root` `=` custom<DynamicIndexList>($root_dynamic, $root) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class ReduceScatterOp(CollectiveCommunicationOperation):
    """
    Reduce-scatter over a device grid.

    Within each device group defined by `grid_axes` reduce the input tensor
    element-wise using the `reduction` method, then scatter the result across
    the group along `scatter_axis`.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardreduce_scatter-shardreducescatterop).
    """

    name = "shard.reduce_scatter"

    input = operand_def(NonZeroRankedTensorConstr)
    reduction = prop_def(
        ReductionKindAttr, default_value=ReductionKindAttr(ReductionKind.SUM)
    )
    scatter_axis = prop_def(IntegerAttr.constr(IndexType))

    result = result_def(TensorType)

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "(`reduction` `=` $reduction^)? "
        + "`scatter_axis` `=` $scatter_axis "
        + "attr-dict `:` type($input) `->` type($result)"
    )

    def verify_(self) -> None:
        super().verify_()
        _verify_same_op_res_rank(self)
        _verify_scatter_or_slice_operand_and_result_shape(
            SSAValue.get(self.input, type=ShapedType),
            self.result,
            self.scatter_axis.value.data,
            self.collective_process_group_size(self.get_grid()),
        )


@irdl_op_definition
class SendOp(CollectiveCommunicationOperation):
    """
    Send from one device to another within a device group.
    """

    name = "shard.send"

    input = operand_def(TensorType)

    destination = prop_def(DenseArrayBase[I64])
    destination_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "`destination` `=` custom<DynamicIndexList>($destination_dynamic, $destination) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class ShiftOp(CollectiveCommunicationOperation):
    """
    Shift over a device shard.

    Within each device group shift along `shift_axis` by `offset`. If the
    `rotate` flag is present a rotation is performed instead of a shift.
    """

    name = "shard.shift"

    input = operand_def(TensorType)

    shift_axis = prop_def(IntegerAttr.constr(IndexType))
    offset = prop_def(IntegerAttr[I64])
    rotate = prop_def(UnitAttr)

    result = result_def(TensorType)

    traits = traits_def(
        Pure(),
    )

    assembly_format = (
        "$input `on` $grid (`grid_axes` `=` $grid_axes^)? "
        + "`shift_axis` `=` $shift_axis "
        + "`offset` `=` $offset "
        + "(`rotate` $rotate^)? "
        + "attr-dict `:` type($input) `->` type($result)"
    )


################################################################################
# Operations on grid                                                           #
################################################################################


@irdl_op_definition
class GridOp(IRDLOperation):
    name = "shard.grid"

    sym_name = prop_def(SymbolNameConstraint())
    shape = prop_def(DenseArrayBase[I64])

    traits = traits_def(SymbolOpInterface(), Pure())

    assembly_format = (
        "$sym_name `(` `shape` `=` custom<DimensionList>($shape) `)` attr-dict"
    )

    custom_directives = (DimensionList,)

    def verify_(self):
        if not self.shape.get_values():
            raise VerifyException(
                "'shard.grid' op rank of grid is expected to be a positive integer"
            )


@irdl_op_definition
class ProcessMultiIndexOp(GridSymbolUserOperation):
    """
    Get the multi index of current device along specified grid axes.

    If `axes` is empty, the index along all grid axes is returned.

    When constructing, pass either the sequence of `axes` (one result per axis),
    or the rank of the grid as an int.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardprocess_multi_index-shardprocessmultiindexop).
    """

    name = "shard.process_multi_index"

    axes = prop_def(ShardAxesAttr, default_value=ShardAxesAttr(i16, BytesAttr(b"")))

    result = var_result_def(IndexType)

    traits = traits_def(Pure())

    assembly_format = "`on` $grid (`axes` `=` $axes^)? attr-dict `:` type($result)"

    def __init__(
        self, grid: str | StringAttr | SymbolRefAttr, axes: Sequence[int] | int
    ):
        if isinstance(axes, int):
            num_results, axes = axes, ()
        else:
            num_results = len(axes)
        super().__init__(
            result_types=[[IndexType()] * num_results],
            properties={
                "grid": SymbolRefAttr.get(grid),
                "axes": ShardAxesAttr.from_list(i16, axes),
            },
        )

    def verify_(self) -> None:
        super().verify_()
        grid = self.get_grid()
        axes = self.axes.get_values()
        _verify_grid_axes(axes, grid)

        expected = len(axes) if axes else len(grid.shape)
        if len(self.result) != expected:
            raise VerifyException(
                f"Unexpected number of results {len(self.result)}. Expected {expected}."
            )


@irdl_op_definition
class ProcessLinearIndexOp(GridSymbolUserOperation):
    """
    Get the linear index of the current device.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardprocess_linear_index-shardprocesslinearindexop).
    """

    name = "shard.process_linear_index"

    result = result_def(IndexType)

    traits = traits_def(Pure())

    assembly_format = "`on` $grid attr-dict `:` type($result)"

    def __init__(self, grid: str | StringAttr | SymbolRefAttr):
        super().__init__(
            result_types=[IndexType()], properties={"grid": SymbolRefAttr.get(grid)}
        )


@irdl_op_definition
class NeighborsLinearIndicesOp(GridSymbolUserOperation):
    """
    For a given grid index, get the linear indices of the direct neighbor
    processes along the given split. A negative value is returned if there is no
    neighbor in the respective direction.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardneighbors_linear_indices-shardneighborslinearindicesop).
    """

    name = "shard.neighbors_linear_indices"

    device = var_operand_def(IndexType)
    split_axes = prop_def(ShardAxesAttr)

    neighbor_down = result_def(IndexType)
    neighbor_up = result_def(IndexType)

    traits = traits_def(Pure())

    assembly_format = (
        "`on` $grid `[` $device `]` `split_axes` `=` $split_axes "
        + "attr-dict `:` type(results)"
    )

    def __init__(
        self,
        grid: str | StringAttr | SymbolRefAttr,
        device: Sequence[SSAValue | Operation],
        split_axes: Sequence[int],
    ):
        super().__init__(
            operands=[device],
            result_types=[IndexType(), IndexType()],
            properties={
                "grid": SymbolRefAttr.get(grid),
                "split_axes": ShardAxesAttr.from_list(i16, split_axes),
            },
        )


################################################################################
# Sharding operations                                                          #
################################################################################


@irdl_op_definition
class ShardingOp(IRDLOperation):
    """
    Define the sharding of a tensor.

    Note: `halo_sizes` and `sharded_dims_offsets` are mutually exclusive.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardsharding-shardshardingop)
    """

    name = "shard.sharding"

    grid = prop_def(FlatSymbolRefAttrConstr)
    split_axes = prop_def(ShardAxesArrayAttr)
    static_sharded_dims_offsets = prop_def(
        DenseArrayBase[I64], default_value=DenseArrayBase[I64](i64, BytesAttr(b""))
    )
    dynamic_sharded_dims_offsets = var_operand_def(I64)
    static_halo_sizes = prop_def(
        DenseArrayBase[I64], default_value=DenseArrayBase[I64](i64, BytesAttr(b""))
    )
    dynamic_halo_sizes = var_operand_def(I64)

    result = result_def(ShardingType)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    traits = traits_def(
        Pure(),
    )

    assembly_format = (
        "$grid "
        + "`split_axes` `=` $split_axes "
        + "(`halo_sizes` `=` custom<DynamicIndexList>($dynamic_halo_sizes, $static_halo_sizes)^)? "
        + "(`sharded_dims_offsets` `=` "
        + "custom<DynamicIndexList>($dynamic_sharded_dims_offsets, $static_sharded_dims_offsets)^)? "
        + "attr-dict `:` type($result)"
    )

    custom_directives = (DynamicIndexList,)

    def verify_(self) -> None:
        dims_offsets = (
            self.static_sharded_dims_offsets or self.dynamic_sharded_dims_offsets
        )
        halo_sizes = self.static_halo_sizes or self.dynamic_halo_sizes

        if dims_offsets and halo_sizes:
            raise VerifyException(
                "'shard.sharding' op halo sizes and shard offsets are mutually exclusive"
            )


@irdl_op_definition
class ShardOp(IRDLOperation):
    """
    Annotate on how a tensor is sharded across a shard.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardshard-shardshardop).
    """

    name = "shard.shard"

    T: ClassVar = VarConstraint("T", TensorType.constr())

    src = operand_def(T)
    sharding = operand_def(ShardingType)
    annotate_for_users = opt_prop_def(UnitAttr)

    result = result_def(T)

    traits = traits_def(
        Pure(),
    )

    assembly_format = "$src `to` $sharding (`annotate_for_users` $annotate_for_users^)? attr-dict `:` type($result)"

    def __init__(
        self,
        src: SSAValue,
        sharding: SSAValue,
        annotate_for_users: UnitAttr | None,
    ):
        return super().__init__(
            operands=[src, sharding],
            result_types=[src.type],
            properties={
                "annotate_for_users": annotate_for_users,
            },
        )


Shard = Dialect(
    "shard",
    [
        AllGatherOp,
        AllReduceOp,
        AllSliceOp,
        BroadcastOp,
        GatherOp,
        RecvOp,
        ReduceOp,
        ReduceScatterOp,
        SendOp,
        ScatterOp,
        ShiftOp,
        GridOp,
        NeighborsLinearIndicesOp,
        ProcessLinearIndexOp,
        ProcessMultiIndexOp,
        ShardingOp,
        ShardOp,
    ],
    [
        ReductionKindAttr,
        ShardingType,
        ShardAxesArrayAttr,
    ],
)
