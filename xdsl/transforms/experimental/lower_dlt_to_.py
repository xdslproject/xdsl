
import abc
import functools
from dataclasses import dataclass
from typing import Optional, cast, assert_type

from xdsl.dialects import llvm, arith, builtin, memref, scf
from xdsl.dialects.builtin import DenseArrayBase, IndexType, UnrealizedConversionCastOp, i64, IntegerType, AnyFloat, \
    IntegerAttr, ModuleOp
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import IndexRangeType, DLTCompatibleElementBaseType
from xdsl.ir import SSAValue, Operation, Attribute, MLContext, Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import RewritePattern, op_type_rewrite_pattern, PatternRewriter, PatternRewriteWalker, \
    GreedyRewritePatternApplier, TypeConversionPattern, attr_type_rewrite_pattern


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
    def get(self) -> tuple[list[Operation], SSAValue] | tuple[None, int]:
        pass


class StaticExtentGetter(ExtentGetter):
    def __init__(self, arg: dlt.StaticExtentAttr):
        assert arg.get_stage() <= dlt.Stage.STATIC
        if isinstance(arg, dlt.StaticExtentAttr):
            self.extent = arg.value.value.data
        else:
            raise NotImplementedError()

    def get(self) -> tuple[None, int]:
        return None, self.extent


class SSAExtentGetter(ExtentGetter):
    def __init__(self, arg: SSAValue):
        self.arg = arg

    def get(self) -> tuple[list[Operation], SSAValue]:
        return [], self.arg


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
        op = llvm.ExtractValueOp(pos, self.input, i64)
        # cast_ops, res = get_as_i64(op.res)
        cast_ops, res = get_as_index(op.res)
        return [op] + cast_ops, res


class ExtentResolver:

    def __init__(self, map: dict[dlt.Extent, ExtentGetter]):
        self.map: dict[dlt.Extent, ExtentGetter] = map
    # def add_bases(self, extent: dlt.Extent, value: int | ExtentGetter):
    #     if isinstance(value, int):
    #         value = SystemExit(extent)
    #     self.map[extent] = value

    def resolve(self, extent: dlt.Extent) -> tuple[list[Operation], SSAValue] | tuple[None, int]:
        if extent.is_static():
            if isinstance(extent, dlt.StaticExtentAttr):
                extent = cast(dlt.StaticExtentAttr, extent)
                return None, extent.as_int()
        if extent in self.map:
            getter = self.map[extent]
            return getter.get()
        else:
            raise KeyError(f"Cannot resolve Extent {extent} in ExtentResolver map {self.map}")


@functools.singledispatch
def get_size_from_layout(layout: dlt.Layout, extent_resolver: ExtentResolver) -> tuple[None, int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    # Returns (Ops, Packed_size, Extra_size)
    # returns the 'Ops' needed to calculate the 'Packed_size' and 'Extra_Size' of the layout
    # This packed size is in general the size of the element assuming it can be tightly packed - contiguously
    # This enables bounded indexing where a pair of indices are used as a start and end of range, and each end is also
    # the next start - the Extra_size then is normally the last end index such that the pointer access
    # range(A[i], A[i+1]) doesn't access unallocated memory
    # all in Bytes
    # Packed_size is int IFF Extra_size is int
    # Packed_size is int IFF Ops is None
    assert isinstance(layout, dlt.Layout)
    raise NotImplementedError(f"Cannot get static llvm type for layout {layout}")


@get_size_from_layout.register
def _(layout: dlt.PrimitiveLayoutAttr, extent_resolver: ExtentResolver) -> tuple[None, int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    if isinstance(layout.base_type, DLTCompatibleElementBaseType):
        p, e = layout.base_type.get_size()
        return None, p, e

    if isinstance(layout.base_type, IntegerType):
        bit_width = layout.base_type.width.data
    elif isinstance(layout.base_type, AnyFloat):
        bit_width = layout.base_type.get_bitwidth
    elif isinstance(layout.base_type, IndexType):
        bit_width = i64.width.data
    else:
        raise ValueError(f"Cannot get size of base element: {layout.base_type}")
    bytes = -(bit_width // -8)
    return None, bytes, 0


@get_size_from_layout.register
def _(layout: dlt.DenseLayoutAttr, extent_resolver: ExtentResolver) -> tuple[None, int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    child_ops, child_size, child_extra = get_size_from_layout(layout.child, extent_resolver)
    extent_ops, extent = extent_resolver.resolve(layout.dimension.extent)

    if child_ops is None and extent_ops is None:
        return None, child_size*extent, child_extra
    child_ops, child_size, child_extra = from_int_to_ssa((child_ops, child_size, child_extra))
    if extent_ops is None:
        extent_ops = [extent_op := arith.Constant(IntegerAttr(extent, IndexType()))]
        extent = extent_op.result
        # cast_ops, extent = get_as_i64(extent_op.result)
        # extent_ops.extend(cast_ops)

    return child_ops + extent_ops + [product := arith.Muli(child_size, extent)], product.result, child_extra


@get_size_from_layout.register
def _(layout: dlt.NamedLayoutAttr, extent_resolver: ExtentResolver) -> tuple[None, int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    return get_size_from_layout(layout.child, extent_resolver)


@get_size_from_layout.register
def _(layout: dlt.MemberLayoutAttr, extent_resolver: ExtentResolver) -> tuple[None, int, int] | tuple[list[Operation], SSAValue, SSAValue]:
    return get_size_from_layout(layout.child, extent_resolver)


def from_int_to_ssa(args: tuple, sum: bool = False) -> tuple:
    """
    This has the type:
    from_int_to_ssa(args: tuple[None, int, ...] | tuple[list[Operation], SSAValue, ...], sum: bool = False) ->
        tuple[list[Operation], SSAValue, ...]

    While this is not standard or easily typable in python the meaning is that the outputs is a tuple, with its first
    element being a list of Operation and all subsequent elements being SSAValue
    """
    ops, *parts = args
    assert isinstance(ops, list | None)
    assert ops is None or all(isinstance(op, Operation) for op in ops)
    assert ops is None or all(isinstance(part, SSAValue) for part in parts)
    assert ops is not None or all(isinstance(part, int) for part in parts)

    if ops is None:
        assert all(isinstance(part, int) for part in parts)
        ops = [arith.Constant(IntegerAttr(part, IndexType())) for part in parts]
        parts = tuple([op.result for op in ops])
    # cast_ops, size = get_as_i64(size)
    # ops.extend(cast_ops)
    # cast_ops, extra = get_as_i64(extra)
    # ops.extend(cast_ops)
    if sum:
        while len(parts)>1:
            new_parts = []
            for i in range(0, len(parts)-1, 2):
                ops.append(add_op := arith.Addi(parts[i], parts[i+1]))
                new_parts.append(add_op.result)
            parts = tuple(new_parts)

    ops = cast(list[Operation], ops)
    parts = cast(tuple[SSAValue], parts)
    return ops, *parts


def get_as_i64(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(value.type, IndexType | IntegerType)
    if isinstance(value.type, IntegerType):
        assert value.type.width.data <= i64.width.data, f"Expected {i64.width.data} got {value.type.width.data}"
    return [op := UnrealizedConversionCastOp.get([value], [i64])], op.outputs[0]


def get_as_index(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(value.type, IndexType | IntegerType)
    if isinstance(value.type, IntegerType):
        assert value.type.width.data <= i64.width.data, f"Expected {i64.width.data} got {value.type.width.data}"
    return [op := UnrealizedConversionCastOp.get([value], [IndexType()])], op.outputs[0]


def get_llvm_type_from_dlt_ptr(ptr_type: dlt.PtrType)->llvm.LLVMStructType:
    return llvm.LLVMStructType.from_type_list(
            [llvm.LLVMPointerType.opaque()]
            + len(ptr_type.filled_dimensions.data) * [i64]
            + len(ptr_type.filled_extents.data) * [i64]
    )


def generate_ptr_struct_in_llvm(output_type: dlt.PtrType, allocated_ptr: SSAValue,
                         dim_map: dict[dlt.DimensionAttr, IndexGetter],
                         extent_resolver: ExtentResolver) -> tuple[list[Operation], SSAValue]:
    ops = [undef_op := llvm.UndefOp(get_llvm_type_from_dlt_ptr(output_type))]
    ptr_struct_result = undef_op.res
    ops.append(set_ptr_op := llvm.InsertValueOp(
        DenseArrayBase.from_list(i64, [0]), ptr_struct_result, allocated_ptr
    ))
    ptr_struct_result = set_ptr_op.res
    idx = 1
    for dim in output_type.filled_dimensions:
        dim_ops, dim_result = dim_map[dim].get()
        ops.extend(dim_ops)
        as_i64_ops, dim_result = get_as_i64(dim_result)
        ops.extend(as_i64_ops)
        ops.append(insert_op := llvm.InsertValueOp(DenseArrayBase.from_list(i64, [idx]), ptr_struct_result, dim_result))
        idx += 1
        ptr_struct_result = insert_op.res
    for extent in output_type.filled_extents:
        extent_ops, extent_result = from_int_to_ssa(extent_resolver.resolve(extent))
        ops.extend(extent_ops)
        as_i64_ops, extent_result = get_as_i64(extent_result)
        ops.extend(as_i64_ops)
        ops.append(
            insert_op := llvm.InsertValueOp(DenseArrayBase.from_list(i64, [idx]), ptr_struct_result, extent_result))
        idx += 1
        ptr_struct_result = insert_op.res
    return ops, ptr_struct_result


@dataclass
class DLTSelectRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, select: dlt.SelectOp, rewriter: PatternRewriter):
        assert isinstance(select.tree.type, dlt.PtrType)
        input_type = cast(dlt.PtrType, select.tree.type)
        input_layout = input_type.layout
        output_type = select.res.type
        assert isinstance(output_type, dlt.PtrType)
        output_type = cast(dlt.PtrType, output_type)
        output_layout = output_type.layout
        print(input_layout, output_layout)

        llvm_in_ptr_type = get_llvm_type_from_dlt_ptr(input_type)
        ops = []
        cast_input_op = UnrealizedConversionCastOp.get(select.tree, llvm_in_ptr_type)
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

        select_ops, select_result = get_select_for(input_layout, output_layout, members, dim_map, extent_resolver, get_ptr_op.res)
        ops.extend(select_ops)

        if input_type.filled_dimensions == output_type.filled_dimensions and input_type.filled_extents == output_type.filled_extents:
            output_llvm_type = llvm_in_ptr_type
            set_ptr_op = llvm.InsertValueOp(
                DenseArrayBase.from_list(i64, [0]), llvm_in, select_result
            )
            ops.append(set_ptr_op)
            ptr_struct_result = set_ptr_op.res
        else:
            gen_ops, ptr_struct_result = generate_ptr_struct_in_llvm(output_type, select_result, dim_map, extent_resolver)
            ops.extend(gen_ops)

        cast_output_op = UnrealizedConversionCastOp.get(ptr_struct_result, output_type)
        ops.append(cast_output_op)
        dlt_out = cast_output_op.outputs[0]

        ops[0].attributes["commentStart"] = builtin.StringAttr("dlt.Select Start")
        ops[-1].attributes["commentEnd"] = builtin.StringAttr("dlt.Select End")
        rewriter.replace_matched_op(ops, [dlt_out])

        pass


@functools.singledispatch
def get_select_for(starting_layout: dlt.Layout, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
    raise NotImplementedError(f"get_select_for not implemented for this layout: {type(starting_layout)} : {starting_layout}")
    # if starting_layout == ending_layout:
    #     return [], input


@get_select_for.register
def _(starting_layout: dlt.PrimitiveLayoutAttr, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert starting_layout == ending_layout
    return [], input

@get_select_for.register
def _(starting_layout: dlt.NamedLayoutAttr, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
    if starting_layout == ending_layout:
        return [], input
    return get_select_for(starting_layout.child, ending_layout, members, dim_mapping, extent_resolver, input)

@get_select_for.register
def _(starting_layout: dlt.MemberLayoutAttr, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert starting_layout.member_specifier in members
    if starting_layout == ending_layout:
        return [], input
    return get_select_for(starting_layout.child, ending_layout, members - {starting_layout.member_specifier}, dim_mapping, extent_resolver, input)

@get_select_for.register
def _(starting_layout: dlt.StructLayoutAttr, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
    if starting_layout == ending_layout:
        return [], input
    index = -1
    child = None
    offset = 0
    offset_ssa = None
    offset_ops = []
    for i, child_layout in enumerate(starting_layout.children):
        child_layout: dlt.Layout = child_layout
        if child_layout.contents_type.has_selectable(members, dim_mapping.keys()):
            index = i
            child = child_layout
            assert not any(c.contents_type.has_selectable(members, dim_mapping.keys()) for _i, c in enumerate(starting_layout.children) if _i > i)
            break
        else:
            ops, size, extra = get_size_from_layout(child_layout, extent_resolver)
            if ops is None:
                offset += size + extra
            else:
                offset_ops.extend(ops)
                offset_ops.append(sum := arith.Addi(size, extra))
                if offset_ssa is not None:
                    offset_ops.append(sum := arith.Addi(offset_ssa, sum))
                    offset_ssa = sum.result
                else:
                    offset_ssa = sum.result
    if offset > 0:
        if offset_ssa is not None:
            offset_ops.append(const_offset := arith.Constant(IntegerAttr(offset, IndexType())))
            offset_ops.append(sum := arith.Addi(offset_ssa, const_offset))
            offset_ssa = sum.result
        else:
            offset_ops.append(const_offset := arith.Constant(IntegerAttr(offset, IndexType())))
            offset_ssa = const_offset.result
    else:
        if offset_ssa is None:
            return get_select_for(child, ending_layout, members, dim_mapping, extent_resolver, input)

    ptr_ops, ptr = add_to_llvm_pointer(input, offset_ssa)
    child_ops, child_res =  get_select_for(child, ending_layout, members, dim_mapping, extent_resolver, ptr)
    return offset_ops + ptr_ops + child_ops, child_res

@get_select_for.register
def _(starting_layout: dlt.DenseLayoutAttr, ending_layout: dlt.Layout, members: set[dlt.MemberAttr], dim_mapping: dict[dlt.DimensionAttr, IndexGetter], extent_resolver: ExtentResolver, input: SSAValue) -> tuple[list[Operation], SSAValue]:
    if starting_layout == ending_layout:
        return [], input
    child_layout = starting_layout.child

    size_ops, size, extra = from_int_to_ssa(get_size_from_layout(child_layout, extent_resolver))

    dim_ops, dim = dim_mapping.pop(starting_layout.dimension).get()
    product_op = arith.Muli(size, dim)

    ptr_ops, ptr = add_to_llvm_pointer(input, product_op.result)

    child_ops, child_res = get_select_for(child_layout, ending_layout, members, dim_mapping, extent_resolver, ptr)
    return size_ops + dim_ops + [product_op] + ptr_ops + child_ops, child_res


@dataclass
class DLTGetRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_op: dlt.GetOp, rewriter: PatternRewriter):
        assert isinstance(get_op.tree.type, dlt.PtrType)
        input_type = cast(dlt.PtrType, get_op.tree.type)
        assert isinstance(input_type, dlt.PtrType)
        get_type = assert_type(get_op.get_type, dlt.AcceptedTypes)
        assert get_op.res.type == get_op.get_type
        assert isinstance(get_type, dlt.AcceptedTypes)

        ops = []
        input_ptr = get_op.tree
        if not isinstance(input_type.layout, dlt.PrimitiveLayoutAttr):
            select_op = dlt.SelectOp(get_op.tree, [], [], [],
                                     input_type.with_new_layout(
                                         dlt.PrimitiveLayoutAttr(base_type=get_type),
                                         remove_bloat=True)
                                     )
            ops.append(select_op)
            input_ptr = select_op.res
            input_type = cast(dlt.PtrType, input_ptr.type)

        cast_input_op = UnrealizedConversionCastOp.get(input_ptr, get_llvm_type_from_dlt_ptr(input_type))
        ops.append(cast_input_op)
        llvm_in = cast_input_op.outputs[0]

        get_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]), llvm_in, llvm.LLVMPointerType.opaque())
        ops.append(get_ptr_op)

        load_op = llvm.LoadOp(get_ptr_op.res, get_type)
        ops.append(load_op)

        rewriter.replace_matched_op(ops, [load_op.dereferenced_value])


@dataclass
class DLTSetRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, set_op: dlt.SetOp, rewriter: PatternRewriter):
        assert isinstance(set_op.tree.type, dlt.PtrType)
        input_type = cast(dlt.PtrType, set_op.tree.type)
        assert isinstance(input_type, dlt.PtrType)
        set_type = assert_type(set_op.set_type, dlt.AcceptedTypes)
        assert set_op.value.type == set_op.set_type
        assert isinstance(set_type, dlt.AcceptedTypes)
        ops = []

        input_ptr = set_op.tree
        if not isinstance(input_type.layout, dlt.PrimitiveLayoutAttr):
            select_op = dlt.SelectOp(set_op.tree, [], [], [],
                                     input_type.with_new_layout(
                                         dlt.PrimitiveLayoutAttr(base_type=set_type),
                                         remove_bloat=True)
                                     )
            ops.append(select_op)
            input_ptr = select_op.res
            input_type = cast(dlt.PtrType, input_ptr.type)

        cast_input_op = UnrealizedConversionCastOp.get(input_ptr, get_llvm_type_from_dlt_ptr(input_type))
        ops.append(cast_input_op)
        llvm_in = cast_input_op.outputs[0]

        get_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]), llvm_in, llvm.LLVMPointerType.opaque())
        ops.append(get_ptr_op)

        store_op = llvm.StoreOp(set_op.value, get_ptr_op.res)
        ops.append(store_op)

        rewriter.replace_matched_op(ops, [])


@dataclass
class DLTAllocRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: dlt.AllocOp, rewriter: PatternRewriter):
        assert isinstance(alloc_op.res.type, dlt.PtrType)
        ptr_type = cast(dlt.PtrType, alloc_op.res.type)

        ops = []

        # input_ptr = set_op.tree
        # if not isinstance(input_type.layout, dlt.PrimitiveLayoutAttr):
        #     select_op = dlt.SelectOp(set_op.tree, [], [], [],
        #                              input_type.with_new_layout(
        #                                  dlt.PrimitiveLayoutAttr(base_type=set_type),
        #                                  remove_bloat=True)
        #                              )
        #     ops.append(select_op)
        #     input_ptr = select_op.res
        #     input_type = cast(dlt.PtrType, input_ptr.type)
        extent_map = {extent: SSAExtentGetter(ssa) for extent, ssa in
                      zip(alloc_op.init_extents, alloc_op.init_extent_sizes)}
        extent_resolver = ExtentResolver(extent_map)

        size_ops, alloc_bytes = from_int_to_ssa(get_size_from_layout(ptr_type.layout, extent_resolver), sum=True)
        ops.extend(size_ops)
        # ops.append(size_sum := arith.Addi(alloc_size, alloc_extra))
        # alloc_bytes = size_sum.result
        ops.append(mr_alloc_op := memref.Alloc.get(IntegerType(8), 64, [-1], [alloc_bytes]))
        ops.append(llvm_alloc_cast := UnrealizedConversionCastOp.get(mr_alloc_op.memref, llvm.LLVMPointerType.opaque()))

        gen_ops, ptr_struct = generate_ptr_struct_in_llvm(ptr_type, llvm_alloc_cast.results[0], {}, extent_resolver)
        ops.extend(gen_ops)


        init_values_map = {cast(dlt.PtrType, init_arg.type).contents_type.get_single_element():init_arg for init_arg in alloc_op.initialValues}

        init_ops = init_layout(ptr_type.layout, extent_resolver, llvm_alloc_cast.results[0], init_values_map)
        ops.extend(init_ops)

        cast_output_op = UnrealizedConversionCastOp.get(ptr_struct, alloc_op.res.type)
        ops.append(cast_output_op)
        dlt_ptr_out = cast_output_op.outputs[0]

        rewriter.replace_matched_op(ops, [dlt_ptr_out])

@functools.singledispatch
def init_layout(layout: dlt.Layout, extent_resolver: ExtentResolver, input_ptr: SSAValue, initial_values: dict[dlt.ElementAttr, SSAValue]) -> list[Operation]:
    assert isinstance(input_ptr.type, llvm.LLVMPointerType)
    assert all(isinstance(initial_value.type, dlt.PtrType) for initial_value in initial_values.values())
    # assert all(element in initial_values for element in layout.contents_type.elements)
    raise NotImplementedError(f"init_layout not implemented for this layout: {type(layout)} : {layout}")

@init_layout.register
def _(layout: dlt.PrimitiveLayoutAttr, extent_resolver: ExtentResolver, input_ptr: SSAValue, initial_values: dict[dlt.ElementAttr, SSAValue]) -> list[Operation]:
    elem = layout.contents_type.get_single_element()
    if elem not in initial_values:
        return []
    return [init_val := dlt.GetOp(initial_values[elem], layout.base_type), llvm.StoreOp(init_val, input_ptr)]


@init_layout.register
def _(layout: dlt.NamedLayoutAttr, extent_resolver: ExtentResolver, input_ptr: SSAValue, initial_values: dict[dlt.ElementAttr, SSAValue]) -> list[Operation]:
    return init_layout(layout.child, extent_resolver, input_ptr, initial_values)


@init_layout.register
def _(layout: dlt.MemberLayoutAttr, extent_resolver: ExtentResolver, input_ptr: SSAValue, initial_values: dict[dlt.ElementAttr, SSAValue]) -> list[Operation]:
    # initial_values = {elem.select_member(layout.member_specifier): val for elem, val in initial_values.items() if elem.has_members([layout.member_specifier])}
    ops = []
    new_init_values = {}
    for elem, dlt_ptr in initial_values.items():
        ops.append(select_op := dlt.SelectOp(dlt_ptr, [layout.member_specifier], [], []))
        new_elem = cast(dlt.PtrType, select_op.res.type).contents_type.get_single_element()
        assert new_elem == elem.select_member(layout.member_specifier)
        new_init_values[new_elem] = select_op.res

    return ops + init_layout(layout.child, extent_resolver, input_ptr, new_init_values)


@init_layout.register
def _(layout: dlt.StructLayoutAttr, extent_resolver: ExtentResolver, input_ptr: SSAValue, initial_values: dict[dlt.ElementAttr, SSAValue]) -> list[Operation]:
    ops = []
    current_input_ptr = input_ptr
    for child in layout.children:
        child = cast(dlt.Layout, child)
        child_initial_values = {elem: val for elem, val in initial_values.items() if
                          elem in child.contents_type}
        ops.extend(init_layout(child, extent_resolver, current_input_ptr, child_initial_values))

        offset_ptr_ops, size = from_int_to_ssa(get_size_from_layout(child, extent_resolver), sum=True)
        ops.extend(offset_ptr_ops)
        increment_ops, current_input_ptr = add_to_llvm_pointer(current_input_ptr, size)
        ops.extend(increment_ops)
    return ops


@init_layout.register
def _(layout: dlt.DenseLayoutAttr, extent_resolver: ExtentResolver, input_ptr: SSAValue, initial_values: dict[dlt.ElementAttr, SSAValue]) -> list[Operation]:
    initial_values = {elem.select_dimension(layout.dimension): val for elem, val in initial_values.items() if elem.has_dimensions([layout.dimension])}
    ops, size, extra = from_int_to_ssa(get_size_from_layout(layout.child, extent_resolver))

    block = Block()
    index = block.insert_arg(IndexType(),0)
    ptr_arg = block.insert_arg(llvm.LLVMPointerType.opaque(), 1)

    new_init_values = {}
    for elem, dlt_ptr in initial_values.items():
        block.add_op(select_op := dlt.SelectOp(dlt_ptr, [], [layout.dimension], [index]))
        new_elem = cast(dlt.PtrType, select_op.res.type).contents_type.get_single_element()
        assert new_elem == elem.select_dimension(layout.dimension)
        new_init_values[new_elem] = select_op.res

    block.add_ops(init_layout(layout.child, extent_resolver, ptr_arg, new_init_values))
    increment_ops, new_ptr = add_to_llvm_pointer(ptr_arg, size)
    block.add_ops(increment_ops)
    block.add_op(scf.Yield(ptr_arg))

    ops.append(lb := arith.Constant(IntegerAttr(0, IndexType())))
    ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))
    ub_ops, ub = from_int_to_ssa(extent_resolver.resolve(layout.dimension.extent))
    ops.extend(ub_ops)
    loop = scf.For(lb, ub, step, [input_ptr], block)
    ops.append(loop)
    return ops



@dataclass
class DLTIterateRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, iterate_op: dlt.IterateOp, rewriter: PatternRewriter):
        assert all(isinstance(tensor.type, dlt.PtrType) for tensor in iterate_op.tensors)
        ops = []


        for tensor_arg, tensor_dims in zip(iterate_op.tensors, iterate_op.dimensions):
            tensor_dims = cast(builtin.ArrayAttr[dlt.SetAttr[dlt.DimensionAttr]], tensor_dims)
            ops.append(cast_op := builtin.UnrealizedConversionCastOp.get(tensor_arg,
                                                                         get_llvm_type_from_dlt_ptr(tensor_arg.type)))
            llvm_tensor_arg = cast_op.outputs[0]
            e_map = {extent: PtrCarriedGetter(llvm_tensor_arg, tensor_arg.type, extent=extent) for extent in
                     tensor_arg.type.filled_extents}
            tensor_arg_extent_resolver = ExtentResolver(e_map)
            for extent, ext_dims in zip(iterate_op.extents, tensor_dims):
                ext_dims = cast(dlt.SetAttr[dlt.DimensionAttr], ext_dims)
                if extent.is_static():
                    assert all(dim.extent == extent for dim in ext_dims)
                else:
                    get_extent_ops, tensor_extent = tensor_arg_extent_resolver.resolve(extent)
                    ops.extend(get_extent_ops)
                    ops.append(cond := arith.Cmpi(extent, tensor_extent, "ne"))
                    fail = [null := llvm.NullOp(llvm.LLVMPointerType.opaque()), llvm.LoadOp(null, llvm.LLVMPointerType.opaque()), scf.Yield()]
                    # fail is a horrible idea to force a seg-fault if the extents don't match.
                    ops.append(scf.If(cond, (), fail, ()))

        extent_map = {extent: SSAExtentGetter(arg) for extent, arg in zip([e for e in iterate_op.extents if not e.is_static()], iterate_op.extent_args)}
        extent_map |= {extent: StaticExtentGetter(extent) for extent in iterate_op.extents if extent.is_static() and isinstance(extent, dlt.StaticExtentAttr)}
        extent_resolver = ExtentResolver(extent_map)

        ops.append(lb := arith.Constant(IntegerAttr(0, IndexType())))
        ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))

        loop_body = Block(arg_types=[IndexType()]+[arg.type for arg in iterate_op.iter_args])
        loop_body.add_op(scf.Yield(*loop_body.args[1:]))
        loop_bodies = [loop_body]
        outer_loop_op = None
        for i, extent in reversed(list(enumerate(iterate_op.extents))):
            extent_ops, ext_ssa = from_int_to_ssa(extent_resolver.resolve(extent))
            ops.extend(extent_ops)
            loop_op = scf.For(lb, ext_ssa, step, iterate_op.iter_args, loop_body)
            if i > 0:
                block = Block(arg_types=[IndexType()]+[arg.type for arg in iterate_op.iter_args])
                block.add_op(loop_op)
                block.add_op(scf.Yield(*loop_op.res))
                loop_body = block
                loop_bodies.append(loop_body)
            else:
                outer_loop_op = loop_op

        indices = list(reversed([body.args[0] for body in loop_bodies]))
        assert len(indices) == len(iterate_op.extents)
        inner_body = loop_bodies[0]

        selectors = []
        block_arg_tensor_types = [arg.type for arg in iterate_op.body.block.args[len(iterate_op.extents):len(iterate_op.extents)+len(iterate_op.tensors)]]
        for tensor_arg, tensor_dims, tensor_type in zip(iterate_op.tensors, iterate_op.dimensions, block_arg_tensor_types):
            tensor_dims = cast(builtin.ArrayAttr[dlt.SetAttr[dlt.DimensionAttr]], tensor_dims)
            dims = []
            values = []
            for index, extent_dims in zip(indices, tensor_dims):
                for dim in extent_dims:
                    dims.append(dim)
                    values.append(index)
            select = dlt.SelectOp(tensor_arg, [], dims, values, tensor_type)
            selectors.append(select)
            rewriter.insert_op_at_start(select, inner_body)

        arg_vals = indices + [s.res for s in selectors] + list(inner_body.args[1:])
        dlt_yield_op = iterate_op.get_yield_op()
        rewriter.inline_block_before(iterate_op.body.block, inner_body.last_op, arg_vals)
        rewriter.replace_op(inner_body.last_op, scf.Yield(*dlt_yield_op.operands))
        rewriter.erase_op(dlt_yield_op)


        ops.append(outer_loop_op)
        rewriter.replace_matched_op(ops, outer_loop_op.results)


@dataclass
class DLTScopeRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, scope_op: dlt.LayoutScopeOp, rewriter: PatternRewriter):
        rewriter.inline_block_before_matched_op(scope_op.body.block)
        rewriter.erase_matched_op()


@dataclass
class DLTPtrTypeRewriter(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: dlt.PtrType, /) -> Attribute | None:
        return get_llvm_type_from_dlt_ptr(typ)


@dataclass
class DLTIndexTypeRewriter(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: builtin.IndexType, /) -> Attribute | None:
        return builtin.i64



def add_to_llvm_pointer(ptr: SSAValue, value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(ptr.type, llvm.LLVMPointerType)
    ops = []
    if isinstance(value.type, IndexType):
        cast_ops, value = get_as_i64(value)
        ops.extend(cast_ops)
    ops.append(ptr_to_int_op := llvm.PtrToIntOp(ptr))
    ops.append(add_op := arith.Addi(ptr_to_int_op.output, value))
    ops.append(int_to_ptr_op := llvm.IntToPtrOp(add_op.result))
    return ops, int_to_ptr_op.output


class LowerDLTPass(ModulePass):
    name = "lwer-dlt"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [DLTSelectRewriter(),
                 DLTGetRewriter(),
                 DLTSetRewriter()]
            )
        )
        walker.rewrite_module(op)
