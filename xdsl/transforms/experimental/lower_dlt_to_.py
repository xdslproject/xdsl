
import abc
import functools
from dataclasses import dataclass
from typing import Optional, cast, assert_type

from xdsl.dialects import llvm, arith
from xdsl.dialects.builtin import DenseArrayBase, IndexType, UnrealizedConversionCastOp, i64, IntegerType, AnyFloat, \
    IntegerAttr
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import IndexRangeType, DLTCompatibleElementBaseType
from xdsl.ir import SSAValue, Operation, Attribute
from xdsl.pattern_rewriter import RewritePattern, op_type_rewrite_pattern, PatternRewriter


class IndexGetter(abc.ABC):
    @abc.abstractmethod
    def get(self) -> tuple[list[Operation], SSAValue]:
        pass


class ArgIndexGetter(IndexGetter):
    def __init__(self, arg: SSAValue):
        self.arg = arg

    def get(self) -> tuple[list[Operation], SSAValue]:
        return [], self.arg


class ExtentGetter(abc.ABC):
    @abc.abstractmethod
    def get(self) -> tuple[list[Operation], int | SSAValue]:
        pass


class StaticExtentGetter(ExtentGetter):
    def __init__(self, arg: dlt.StaticExtentAttr):
        assert arg.get_stage() <= dlt.Stage.STATIC
        if isinstance(arg, dlt.StaticExtentAttr):
            self.extent = arg.value.value.data
        else:
            raise NotImplementedError()

    def get(self) -> tuple[list[Operation], int]:
        return [], self.extent


class PtrCarriedGetter(IndexGetter, ExtentGetter):
    def __init__(self, input: SSAValue, ptr_type: dlt.PtrType, *, dim: dlt.DimensionAttr = None, extent: dlt.Extent = None):
        self.input = input
        self.ptr_type = ptr_type
        self.dim = dim
        self.extent = extent
        assert (self.dim is None) ^ (self.extent is None)

    def get(self) -> tuple[list[Operation], SSAValue]:
        if self.dim is not None:
            index = 1 + self.ptr_type.filled_dimensions.data.index(self.dim)
        elif self.extent is not None:
            index = 1 + len(self.ptr_type.filled_dimensions) + self.ptr_type.filled_extents.data.index(self.extent)
        else:
            assert False
        pos = DenseArrayBase.from_list(i64, [index])
        op = llvm.ExtractValueOp(pos, self.input, IndexType())
        return [op], op.res

def get_llvm_type_from_dlt_ptr(ptr_type: dlt.PtrType)->llvm.LLVMStructType:
    return llvm.LLVMStructType.from_type_list(
            [llvm.LLVMPointerType.opaque()]
            + len(ptr_type.filled_dimensions.data) * [IndexType()]
            + len(ptr_type.filled_extents.data) * [IndexType()]
    )

class ExtentResolver:

    def __init__(self, map: dict[dlt.Extent, ExtentGetter]):
        self.map: dict[dlt.Extent, ExtentGetter] = map
    # def add_bases(self, extent: dlt.Extent, value: int | ExtentGetter):
    #     if isinstance(value, int):
    #         value = SystemExit(extent)
    #     self.map[extent] = value

    def resolve(self, extent: dlt.Extent) -> tuple[list[Operation], int | SSAValue]:
        if extent.is_static():
            if isinstance(extent, dlt.StaticExtentAttr):
                extent = cast(dlt.StaticExtentAttr, extent)
                return [], extent.as_int()
        if extent in self.map:
            getter = self.map[extent]
            return getter.get()
        else:
            raise KeyError(f"Cannot resolve Extent {extent} in ExtentResolver map {self.map}")



@functools.singledispatch
def get_size_from_layout(layout: dlt.Layout, extent_resolver: ExtentResolver) -> tuple[list[Operation], int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    # Returns (Ops, Packed_size, Extra_size)
    # returns the 'Ops' needed to calculate the 'Packed_size' and 'Extra_Size' of the layout
    # This packed size is in general the size of the element assuming it can be tightly packed - contiguously
    # This enables bounded indexing where a pair of indices are used as a start and end of range, and each end is also
    # the next start - the Extra_size then is normally the last end index such that the pointer access
    # range(A[i], A[i+1]) doesn't access unallocated memory
    # all in Bytes
    # Packed_size is int IFF Extra_size is int
    # Packed_size is int IFF len(Ops) == 0
    assert isinstance(layout, dlt.Layout)
    raise NotImplementedError(f"Cannot get static llvm type for layout {layout}")


@get_size_from_layout.register
def _(layout: dlt.PrimitiveLayoutAttr, extent_resolver: ExtentResolver) -> tuple[list[Operation], int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    if isinstance(layout.base_type, DLTCompatibleElementBaseType):
        p, e = layout.base_type.get_size()
        return [], p, e

    if isinstance(layout.base_type, IntegerType):
        bit_width = layout.base_type.width.data
    elif isinstance(layout.base_type, AnyFloat):
        bit_width = layout.base_type.get_bitwidth
    elif isinstance(layout.base_type, IndexType):
        bit_width = i64.width.data
    else:
        raise ValueError(f"Cannot get size of base element: {layout.base_type}")
    bytes = -(bit_width // -8)
    return [], bytes, 0


@get_size_from_layout.register
def _(layout: dlt.DenseLayoutAttr, extent_resolver: ExtentResolver) -> tuple[list[Operation], int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    child_ops, child_size, child_extra = get_size_from_layout(layout.child, extent_resolver)
    extent_ops, extent = extent_resolver.resolve(layout.dimension.extent)

    if isinstance(child_size, int) and isinstance(extent, int):
        assert len(extent_ops) == 0
        assert len(child_ops) == 0
        return [], child_size*extent, child_extra
    child_ops, child_size, child_extra = from_int_to_ssa((child_ops, child_size, child_extra))
    if isinstance(extent, int):
        extent_ops.append(extent_op := arith.Constant(IntegerAttr(extent, IndexType())))
        extent = extent_op.result
    return child_ops + extent_ops + [product := arith.Muli(child_size, extent)], product.result, child_extra


@get_size_from_layout.register
def _(layout: dlt.NamedLayoutAttr, extent_resolver: ExtentResolver) -> tuple[list[Operation], int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    return get_size_from_layout(layout.child, extent_resolver)


def from_int_to_ssa(args: tuple[list[Operation], int, int] | tuple[list[Operation], SSAValue, SSAValue]) -> tuple[list[Operation], SSAValue, SSAValue]:
    ops, size, extra = args
    if isinstance(size, int):
        assert isinstance(extra, int)
        ops.append(size_const := arith.Constant(IntegerAttr(size, IndexType())))
        size = size_const.result
        ops.append(extra_const := arith.Constant(IntegerAttr(extra, IndexType())))
        extra = extra_const.result
    return ops, size, extra



@dataclass
class DLTSelectRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, select: dlt.SelectOp, rewriter: PatternRewriter):
        input_type: dlt.PtrType = select.tree.type
        assert isinstance(input_type, dlt.PtrType)
        input_layout = input_type.layout
        output_type = select.res.type
        assert isinstance(output_type, dlt.PtrType)
        output_layout = output_type.layout
        print(input_layout, output_layout)

        llvm_in_ptr_type = get_llvm_type_from_dlt_ptr(input_type)
        ops = []
        cast_input_op = UnrealizedConversionCastOp(select.tree, llvm_in_ptr_type)
        ops.append(cast_input_op)
        llvm_in = cast_input_op.outputs[0]


        members = set(input_type.filled_members) | set(select.members)
        dim_map = {dim:ArgIndexGetter(val) for dim, val in zip(select.dimensions, select.values)}
        dim_map |= {dim: PtrCarriedGetter(llvm_in, input_type, dim=dim) for dim in input_type.filled_dimensions}

        extent_map = {extent:PtrCarriedGetter(llvm_in, input_type, extent=extent) for extent in input_type.filled_extents}
        extent_resolver = ExtentResolver(extent_map)

        get_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]), llvm_in, llvm.LLVMPointerType.opaque())
        ops.append(get_ptr_op)

        select_ops, result = self.get_select_for(input_layout, output_layout, members, dim_map, extent_resolver, get_ptr_op.res)
        ops.extend(select_ops)

        cast_output_op = UnrealizedConversionCastOp(result, output_type)
        ops.append(cast_output_op)
        dlt_out = cast_output_op.outputs[0]

        rewriter.replace_matched_op(ops, [dlt_out])

        pass

    @op_type_rewrite_pattern
    def match_and_rewrite(self, get: dlt.GetOp, rewriter: PatternRewriter):
        assert isinstance(get.tree.type, dlt.PtrType)
        input_type = cast(dlt.PtrType, get.tree.type)
        assert isinstance(input_type, dlt.PtrType)
        input_layout = input_type.layout
        get_type = assert_type(get.get_type, dlt.AcceptedTypes)
        assert get.res.type == get.get_type
        assert isinstance(get_type, dlt.AcceptedTypes)

        ops = []
        cast_input_op = UnrealizedConversionCastOp.get(get.tree, get_llvm_type_from_dlt_ptr(input_type))
        ops.append(cast_input_op)
        llvm_in = cast_input_op.outputs[0]

        members = set(input_type.filled_members)
        dim_map = {dim: PtrCarriedGetter(llvm_in, input_type, dim=dim) for dim in input_type.filled_dimensions}

        extent_map = {extent: PtrCarriedGetter(llvm_in, input_type, extent=extent) for extent in
                      input_type.filled_extents}
        extent_resolver = ExtentResolver(extent_map)


        get_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]), llvm_in, llvm.LLVMPointerType.opaque())
        ops.append(get_ptr_op)

        select_ops, result = self.get_select_for(input_layout, dlt.PrimitiveLayoutAttr(base_type=get_type), members, dim_map, extent_resolver, get_ptr_op.res)
        ops.extend(select_ops)

        assert isinstance(result.type, llvm.LLVMPointerType)
        load_op = llvm.LoadOp(result, get_type)
        ops.append(load_op)

        rewriter.replace_matched_op(ops, [load_op.dereferenced_value])

    @functools.singledispatchmethod
    def get_select_for(self, starting_layout: dlt.Layout, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
        raise NotImplementedError(f"get_select_for not implemented for this layout: {type(starting_layout)} : {starting_layout}")
        # if starting_layout == ending_layout:
        #     return [], input

    @get_select_for.register
    def _(self, starting_layout: dlt.PrimitiveLayoutAttr, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
        assert starting_layout == ending_layout
        return [], input

    @get_select_for.register
    def _(self, starting_layout: dlt.NamedLayoutAttr, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
        if starting_layout == ending_layout:
            return [], input
        return self.get_select_for(starting_layout.child, ending_layout, members, dim_mapping, extent_resolver, input)

    @get_select_for.register
    def _(self, starting_layout: dlt.DenseLayoutAttr, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
        if starting_layout == ending_layout:
            return [], input
        child_layout = starting_layout.child

        size_ops, size, extra = from_int_to_ssa(get_size_from_layout(child_layout, extent_resolver))
        dim_ops, dim = dim_mapping[starting_layout.dimension].get()

        ptr_to_int_op = llvm.PtrToIntOp(input)
        product_op = arith.Muli(size, dim)
        add_op = arith.Addi(ptr_to_int_op.output, product_op.result)
        int_to_ptr_op = llvm.IntToPtrOp(add_op.result)

        child_ops, child_res = self.get_select_for(child_layout, ending_layout, members, dim_mapping, extent_resolver, int_to_ptr_op.output)
        return size_ops + dim_ops + [ptr_to_int_op, product_op, add_op, int_to_ptr_op] + child_ops, child_res