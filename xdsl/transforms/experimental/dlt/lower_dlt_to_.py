from dataclasses import dataclass
from typing import assert_type, cast

from xdsl.dialects import arith, builtin, llvm, printf, scf
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    UnrealizedConversionCastOp,
    i64,
)
from xdsl.dialects.experimental import dlt
from xdsl.ir import Attribute, Block, BlockArgument, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.transforms.experimental.dlt import layout_llvm_semantics
from xdsl.transforms.experimental.dlt.layout_llvm_semantics import (
    ArgIndexGetter,
    ExtentResolver,
    IndexGetter,
    NoCallback,
    PtrCarriedExtentGetter,
    PtrCarriedIndexGetter,
    SSAExtentGetter,
    Semantic_Map,
    ValueMapInitialiser,
)


def get_as_i64(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(value.type, IndexType | IntegerType)
    if isinstance(value.type, IntegerType):
        assert (
            value.type.width.data <= i64.width.data
        ), f"Expected {i64.width.data} got {value.type.width.data}"
    return [op := UnrealizedConversionCastOp.get([value], [i64])], op.outputs[0]


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
        # print(input_layout, output_layout)

        llvm_in_ptr_type = Semantic_Map.get_data_type_from_dlt_ptr(input_type)
        ops = []
        cast_input_op = UnrealizedConversionCastOp.get(select.tree, llvm_in_ptr_type)
        ops.append(cast_input_op)
        llvm_in = cast_input_op.outputs[0]

        members = set(input_type.filled_members) | set(select.members)
        dim_map = {
            dim: ArgIndexGetter(val)
            for dim, val in zip(select.dimensions, select.values)
        }
        dim_map |= {
            dim: PtrCarriedIndexGetter(llvm_in, input_type, dim=dim)
            for dim in input_type.filled_dimensions
        }

        extent_map = {
            extent: PtrCarriedExtentGetter(llvm_in, input_type, extent=extent)
            for extent in input_type.filled_extents
        }
        extent_resolver = ExtentResolver(extent_map, select.get_scope())

        get_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]), llvm_in, llvm.LLVMPointerType.opaque()
        )
        ops.append(get_ptr_op)

        select_ops, select_result = Semantic_Map.get_select_for(
            input_layout,
            output_layout,
            members,
            dim_map,
            extent_resolver,
            get_ptr_op.res,
        )
        ops.extend(select_ops)

        if (
            input_type.filled_dimensions == output_type.filled_dimensions
            and input_type.filled_extents == output_type.filled_extents
        ):
            output_llvm_type = llvm_in_ptr_type
            set_ptr_op = llvm.InsertValueOp(
                DenseArrayBase.from_list(i64, [0]), llvm_in, select_result
            )
            ops.append(set_ptr_op)
            ptr_struct_result = set_ptr_op.res
        else:
            gen_ops, ptr_struct_result = Semantic_Map.generate_ptr_struct(
                output_type, select_result, dim_map, extent_resolver
            )
            ops.extend(gen_ops)

        cast_output_op = UnrealizedConversionCastOp.get(ptr_struct_result, output_type)
        ops.append(cast_output_op)
        dlt_out = cast_output_op.outputs[0]

        ops[0].attributes["commentStart"] = builtin.StringAttr("dlt.Select Start")
        ops[-1].attributes["commentEnd"] = builtin.StringAttr("dlt.Select End")
        rewriter.replace_matched_op(ops, [dlt_out])

        pass


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
        llvm_in_ptr_type = Semantic_Map.get_data_type_from_dlt_ptr(input_type)
        cast_input_op = UnrealizedConversionCastOp.get(get_op.tree, llvm_in_ptr_type)
        ops.append(cast_input_op)
        llvm_dlt_ptr_in = cast_input_op.outputs[0]

        get_data_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]),
            llvm_dlt_ptr_in,
            llvm.LLVMPointerType.opaque(),
        )
        ops.append(get_data_ptr_op)
        llvm_data_ptr = get_data_ptr_op.res

        get_ops, get_res, get_found = Semantic_Map.get_getter_for(
            input_type.layout,
            get_type,
            set(input_type.filled_members),
            {
                dim: PtrCarriedIndexGetter(llvm_dlt_ptr_in, input_type, dim=dim)
                for dim in input_type.filled_dimensions
            },
            ExtentResolver(
                {
                    extent: PtrCarriedExtentGetter(
                        llvm_dlt_ptr_in, input_type, extent=extent
                    )
                    for extent in input_type.filled_extents
                },
                get_op.get_scope()
            ),
            llvm_data_ptr,
        )
        ops.extend(get_ops)
        if get_found is not True:
            zero_ops, zero_val = layout_llvm_semantics._get_packed_zero_for_accepted_type(get_type)
            ops.extend(zero_ops)
            if isinstance(get_found, SSAValue):
                ops.append(select_op := arith.Select(get_found, get_res, zero_val))
                get_res = select_op.result
            else:
                get_res = zero_val


        rewriter.replace_matched_op(ops, [get_res])


@dataclass
class DLTGetSRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_op: dlt.GetSOp, rewriter: PatternRewriter):
        assert isinstance(get_op.tree.type, dlt.PtrType)
        input_type = cast(dlt.PtrType, get_op.tree.type)
        assert isinstance(input_type, dlt.PtrType)
        get_type = assert_type(get_op.get_type, dlt.AcceptedTypes)
        assert get_op.res.type == get_op.get_type
        assert isinstance(get_type, dlt.AcceptedTypes)

        ops = []
        llvm_in_ptr_type = Semantic_Map.get_data_type_from_dlt_ptr(input_type)
        cast_input_op = UnrealizedConversionCastOp.get(get_op.tree, llvm_in_ptr_type)
        ops.append(cast_input_op)
        llvm_dlt_ptr_in = cast_input_op.outputs[0]

        get_data_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]),
            llvm_dlt_ptr_in,
            llvm.LLVMPointerType.opaque(),
        )
        ops.append(get_data_ptr_op)
        llvm_data_ptr = get_data_ptr_op.res

        get_ops, get_res, get_found = Semantic_Map.get_getter_for(
            input_type.layout,
            get_type,
            set(input_type.filled_members),
            {
                dim: PtrCarriedIndexGetter(llvm_dlt_ptr_in, input_type, dim=dim)
                for dim in input_type.filled_dimensions
            },
            ExtentResolver(
                {
                    extent: PtrCarriedExtentGetter(
                        llvm_dlt_ptr_in, input_type, extent=extent
                    )
                    for extent in input_type.filled_extents
                },
                get_op.get_scope()
            ),
            llvm_data_ptr,
        )
        ops.extend(get_ops)

        bool_ops, get_found = layout_llvm_semantics._make_bool_ssa(get_found)
        ops.extend(bool_ops)

        zero_ops, zero_val = layout_llvm_semantics._get_packed_zero_for_accepted_type(get_type)
        ops.extend(zero_ops)

        ops.append(select_op := arith.Select(get_found, get_res, zero_val))
        gets_res = select_op.result
        rewriter.replace_matched_op(ops, [gets_res, get_found])


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
        llvm_in_ptr_type = Semantic_Map.get_data_type_from_dlt_ptr(input_type)
        cast_input_op = UnrealizedConversionCastOp.get(set_op.tree, llvm_in_ptr_type)
        ops.append(cast_input_op)
        llvm_dlt_ptr_in = cast_input_op.outputs[0]

        get_data_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]),
            llvm_dlt_ptr_in,
            llvm.LLVMPointerType.opaque(),
        )
        ops.append(get_data_ptr_op)
        llvm_data_ptr = get_data_ptr_op.res

        set_ops = Semantic_Map.get_setter_for(
            input_type.layout,
            set_op.value,
            set(input_type.filled_members),
            {
                dim: PtrCarriedIndexGetter(llvm_dlt_ptr_in, input_type, dim=dim)
                for dim in input_type.filled_dimensions
            },
            ExtentResolver(
                {
                    extent: PtrCarriedExtentGetter(
                        llvm_dlt_ptr_in, input_type, extent=extent
                    )
                    for extent in input_type.filled_extents
                },
                set_op.get_scope()
            ),
            llvm_data_ptr,
        )
        ops.extend(set_ops)

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
        extent_map = {
            extent: SSAExtentGetter(ssa)
            for extent, ssa in zip(alloc_op.init_extents, alloc_op.init_extent_sizes)
        }
        extent_resolver = ExtentResolver(extent_map, alloc_op.get_scope())

        # size_ops, alloc_bytes = from_int_to_ssa(
        #     get_size_from_layout(ptr_type.layout, extent_resolver), sum=True
        # )
        size_ops, (alloc_bytes,) = (
            Semantic_Map.get_size(ptr_type.layout, extent_resolver).sum().output()
        )
        ops.extend(size_ops)

        # ops.append(mr_alloc_op := memref.Alloc.get(IntegerType(8), 64, [-1], [alloc_bytes]))
        # ops.append(llvm_alloc_cast := UnrealizedConversionCastOp.get(mr_alloc_op.memref, llvm.LLVMPointerType.opaque()))
        # buffer = llvm_alloc_cast.results[0]

        conv_to_i64, alloc_bytes = get_as_i64(alloc_bytes)
        ops.extend(conv_to_i64)
        ops.append(
            malloc := llvm.CallOp(
                "malloc", alloc_bytes, return_type=llvm.LLVMPointerType.opaque()
            )
        )
        buffer = malloc.returned

        gen_ops, ptr_struct = Semantic_Map.generate_ptr_struct(
            ptr_type, buffer, {}, extent_resolver
        )
        ops.extend(gen_ops)

        init_values_map = {
            cast(
                dlt.PtrType, init_arg.type
            ).contents_type: init_arg
            for init_arg in alloc_op.initialValues
        }
        value_map_initialiser = ValueMapInitialiser(Semantic_Map, extent_resolver, init_values_map)

        # init_ops = init_layout(
        #     ptr_type.layout, extent_resolver, buffer, init_values_map
        # )
        init_ops, callback_ret = Semantic_Map.init_layout(
            ptr_type.layout,
            extent_resolver,
            buffer,
            value_map_initialiser,
            NoCallback(),
            [],
            True,
            True,
        )
        ops.extend(init_ops)

        cast_output_op = UnrealizedConversionCastOp.get(ptr_struct, alloc_op.res.type)
        ops.append(cast_output_op)
        dlt_ptr_out = cast_output_op.outputs[0]

        rewriter.replace_matched_op(ops, [dlt_ptr_out])



@dataclass
class DLTDeallocRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, dealloc_op: dlt.DeallocOp, rewriter: PatternRewriter):
        assert isinstance(dealloc_op.tree.type, dlt.PtrType)
        ptr_type = cast(dlt.PtrType, dealloc_op.tree.type)

        ops = []

        llvm_in_ptr_type = Semantic_Map.get_data_type_from_dlt_ptr(ptr_type)
        cast_input_op = UnrealizedConversionCastOp.get(dealloc_op.tree, llvm_in_ptr_type)
        ops.append(cast_input_op)
        llvm_in = cast_input_op.outputs[0]

        extent_map = {
            extent: PtrCarriedExtentGetter(llvm_in, ptr_type, extent=extent)
            for extent in ptr_type.filled_extents
        }
        extent_resolver = ExtentResolver(extent_map, dealloc_op.get_scope())

        get_data_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [0]),
            llvm_in,
            llvm.LLVMPointerType.opaque(),
        )
        ops.append(get_data_ptr_op)
        llvm_data_ptr = get_data_ptr_op.res

        dealloc_ops = Semantic_Map.dealloc_layout(
            ptr_type.layout,
            extent_resolver,
            llvm_data_ptr,
        )

        ops.extend(dealloc_ops)

        ops.append(
            free := llvm.CallOp(
                "free", llvm_data_ptr, return_type=None
            )
        )

        rewriter.replace_matched_op(ops, [])

# @functools.singledispatch
# def init_layout(
#     layout: dlt.Layout,
#     extent_resolver: ExtentResolver,
#     input_ptr: SSAValue,
#     initial_values: dict[dlt.ElementAttr, SSAValue],
# ) -> list[Operation]:
#     assert isinstance(input_ptr.type, llvm.LLVMPointerType)
#     assert all(
#         isinstance(initial_value.type, dlt.PtrType)
#         for initial_value in initial_values.values()
#     )
#     # assert all(element in initial_values for element in layout.contents_type.elements)
#     raise NotImplementedError(
#         f"init_layout not implemented for this layout: {type(layout)} : {layout}"
#     )

#
# @init_layout.register
# def _(
#     layout: dlt.PrimitiveLayoutAttr,
#     extent_resolver: ExtentResolver,
#     input_ptr: SSAValue,
#     initial_values: dict[dlt.ElementAttr, SSAValue],
# ) -> list[Operation]:
#     elem = layout.contents_type.get_single_element()
#     debug = (
#         []
#     )  # [int_op := llvm.PtrToIntOp(input_ptr), trunc_op := arith.TruncIOp(int_op.output, builtin.i32), printf.PrintIntOp(trunc_op.result)]
#     if elem not in initial_values:
#         return debug + []
#     return debug + [
#         init_val := dlt.GetOp(initial_values[elem], layout.base_type),
#         llvm.StoreOp(init_val, input_ptr),
#     ]
#
#
# # @init_layout.register
# # def _(
# #     layout: dlt.NamedLayoutAttr,
# #     extent_resolver: ExtentResolver,
# #     input_ptr: SSAValue,
# #     initial_values: dict[dlt.ElementAttr, SSAValue],
# # ) -> list[Operation]:
# #     return init_layout(layout.child, extent_resolver, input_ptr, initial_values)
#
#
# @init_layout.register
# def _(
#     layout: dlt.MemberLayoutAttr,
#     extent_resolver: ExtentResolver,
#     input_ptr: SSAValue,
#     initial_values: dict[dlt.ElementAttr, SSAValue],
# ) -> list[Operation]:
#     # initial_values = {elem.select_member(layout.member_specifier): val for elem, val in initial_values.items() if elem.has_members([layout.member_specifier])}
#     ops = []
#     new_init_values = {}
#     for elem, dlt_ptr in initial_values.items():
#         ops.append(
#             select_op := dlt.SelectOp(dlt_ptr, [layout.member_specifier], [], [])
#         )
#         new_elem = cast(
#             dlt.PtrType, select_op.res.type
#         ).contents_type.get_single_element()
#         assert new_elem == elem.select_member(layout.member_specifier)
#         new_init_values[new_elem] = select_op.res
#
#     return ops + init_layout(layout.child, extent_resolver, input_ptr, new_init_values)
#
#
# @init_layout.register
# def _(
#     layout: dlt.StructLayoutAttr,
#     extent_resolver: ExtentResolver,
#     input_ptr: SSAValue,
#     initial_values: dict[dlt.ElementAttr, SSAValue],
# ) -> list[Operation]:
#     ops = []
#     current_input_ptr = input_ptr
#     for child in layout.children:
#         child = cast(dlt.Layout, child)
#         child_initial_values = {
#             elem: val
#             for elem, val in initial_values.items()
#             if elem in child.contents_type
#         }
#         ops.extend(
#             init_layout(child, extent_resolver, current_input_ptr, child_initial_values)
#         )
#
#         offset_ptr_ops, size = from_int_to_ssa(
#             get_size_from_layout(child, extent_resolver), sum=True
#         )
#         ops.extend(offset_ptr_ops)
#         increment_ops, current_input_ptr = add_to_llvm_pointer(current_input_ptr, size)
#         ops.extend(increment_ops)
#     return ops
#
#
# @init_layout.register
# def _(
#     layout: dlt.DenseLayoutAttr,
#     extent_resolver: ExtentResolver,
#     input_ptr: SSAValue,
#     initial_values: dict[dlt.ElementAttr, SSAValue],
# ) -> list[Operation]:
#     initial_values = {
#         elem.select_dimension(layout.dimension): val
#         for elem, val in initial_values.items()
#         if elem.has_dimensions([layout.dimension])
#     }
#     ops, size, extra = from_int_to_ssa(
#         get_size_from_layout(layout.child, extent_resolver)
#     )
#
#     block = Block()
#     index = block.insert_arg(IndexType(), 0)
#     block.add_op(offsetMul_op := arith.Muli(index, size))
#     ptr_add_ops, ptr_arg = add_to_llvm_pointer(input_ptr, offsetMul_op.result)
#     block.add_ops(ptr_add_ops)
#
#     new_init_values = {}
#     for elem, dlt_ptr in initial_values.items():
#         block.add_op(
#             select_op := dlt.SelectOp(dlt_ptr, [], [layout.dimension], [index])
#         )
#         new_elem = cast(
#             dlt.PtrType, select_op.res.type
#         ).contents_type.get_single_element()
#         assert new_elem == elem.select_dimension(layout.dimension)
#         new_init_values[new_elem] = select_op.res
#
#     block.add_ops(init_layout(layout.child, extent_resolver, ptr_arg, new_init_values))
#     # block.add_ops(increment_ops)
#     block.add_op(scf.Yield())
#
#     ops.append(lb := arith.Constant(IntegerAttr(0, IndexType())))
#     ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))
#     ub_ops, ub = from_int_to_ssa(extent_resolver.resolve(layout.dimension.extent))
#     ops.extend(ub_ops)
#     debug = [
#         trunc_op := arith.IndexCastOp(ub, builtin.i32),
#         printf.PrintIntOp(trunc_op.result),
#     ]
#     # ops.extend(debug)
#     loop = scf.For(lb, ub, step, [], block)
#     ops.append(loop)
#     return ops


@dataclass
class DLTIterateRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, iterate_op: dlt.IterateOp, rewriter: PatternRewriter):
        assert all(
            isinstance(tensor.type, dlt.PtrType) for tensor in iterate_op.tensors
        )
        ops = []

        extent_map = {
            extent: SSAExtentGetter(arg)
            for extent, arg in zip(
                [e for e in iterate_op.extents if e.get_stage() >= dlt.Stage.INIT],
                iterate_op.extent_args,
            )
        }
        # extent_map |= {
        #     extent: StaticExtentGetter(extent)
        #     for extent in iterate_op.extents
        #     if extent.is_static() and isinstance(extent, dlt.StaticExtentAttr)
        # }
        extent_resolver = ExtentResolver(extent_map, iterate_op.get_scope())

        for tensor_arg, tensor_dims in zip(iterate_op.tensors, iterate_op.dimensions):
            tensor_dims = cast(
                builtin.ArrayAttr[dlt.SetAttr[dlt.DimensionAttr]], tensor_dims
            )
            ops.append(
                cast_op := builtin.UnrealizedConversionCastOp.get(
                    tensor_arg, Semantic_Map.get_data_type_from_dlt_ptr(tensor_arg.type)
                )
            )
            llvm_tensor_arg = cast_op.outputs[0]
            e_map = {
                extent: PtrCarriedExtentGetter(
                    llvm_tensor_arg, tensor_arg.type, extent=extent
                )
                for extent in tensor_arg.type.filled_extents
            }
            tensor_arg_extent_resolver = ExtentResolver(e_map, iterate_op.get_scope())
            for extent, ext_dims in zip(iterate_op.extents, tensor_dims):
                ext_dims = cast(dlt.SetAttr[dlt.DimensionAttr], ext_dims)
                if extent.is_static():
                    assert all(dim.extent == extent for dim in ext_dims)
                elif extent.is_scope_time():
                    assert all(dim.extent == extent for dim in ext_dims)
                else:
                    get_tensor_extent_ops, (tensor_extent,) = (
                        tensor_arg_extent_resolver.resolve(extent).output()
                    )

                    ops.extend(get_tensor_extent_ops)
                    get_iterate_extent_ops, (iterate_extent,) = extent_resolver.resolve(
                        extent
                    ).output()
                    ops.extend(get_iterate_extent_ops)
                    ops.append(cond := arith.Cmpi(iterate_extent, tensor_extent, "ne"))
                    fail = [
                        printf.PrintFormatOp(
                            f"Failed Extent Assertion for {extent}: {{}} != {{}} in Iterate {iterate_op.identification.data} over {[e for e in iterate_op.extents]}",
                            iterate_extent,
                            tensor_extent,
                        ),
                        llvm.CallOp("abort"),
                        scf.Yield(),
                    ]
                    ops.append(scf.If(cond, (), fail))

        ops.append(lb := arith.Constant(IntegerAttr(0, IndexType())))
        ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))

        def _make_inner_body(indices_map: dict[int, BlockArgument], iter_args: list[SSAValue], insert_point: InsertPoint) -> list[SSAValue]:
            _inner_body_ops = []

            selected_tensors = []
            block_arg_tensor_types = [arg.type for arg in iterate_op.get_block_args_for_tensor_args()]
            indices = [indices_map[i] for i in range(len(iterate_op.extents))]

            for tensor_arg, tensor_dims, tensor_type in zip(
                    iterate_op.tensors, iterate_op.dimensions, block_arg_tensor_types
            ):
                tensor_dims = cast(
                    builtin.ArrayAttr[dlt.SetAttr[dlt.DimensionAttr]], tensor_dims
                )
                dims = []
                values = []
                for index, extent_dims in zip(indices, tensor_dims):
                    for dim in extent_dims:
                        dims.append(dim)
                        values.append(index)
                select = dlt.SelectOp(tensor_arg, [], dims, values, tensor_type)
                selected_tensors.append(select.res)
                _inner_body_ops.append(select)

            iterate_op_body_arg_vals = indices + selected_tensors + iter_args
            dlt_yield_op = iterate_op.get_yield_op()
            yielded = dlt_yield_op.arguments
            dlt_yield_op.detach()
            dlt_yield_op.erase()
            Rewriter.insert_ops_at_location(_inner_body_ops, insert_point)
            Rewriter.inline_block_at_location(iterate_op.body.block, insert_point, iterate_op_body_arg_vals)

            return yielded

        def _make_for_loop(iteration_order: dlt.IterationOrder, iter_args: list[SSAValue], indices_map: dict[int, BlockArgument], insert_point: InsertPoint) -> list[SSAValue]:
            if isinstance(iteration_order, dlt.BodyIterationOrderAttr):
                resulting_iter_args = _make_inner_body(indices_map, iter_args, insert_point)
                return resulting_iter_args
            elif isinstance(iteration_order, dlt.NestedIterationOrderAttr):
                extent_idx = iteration_order.extent_index.data
                extent = iterate_op.extents.data[extent_idx]
                extent_ops, (ext_ssa,) = extent_resolver.resolve(extent).output()
                ops.extend(extent_ops)
                loop_body = Block(
                    arg_types=[IndexType()] + [arg.type for arg in iter_args]
                )
                sub_iter_args = list(loop_body.args[1:])
                sub_indices_map = indices_map | {extent_idx: loop_body.args[0]}
                sub_insert_point = InsertPoint.at_end(loop_body)
                resulting_iter_args = _make_for_loop(iteration_order.child, sub_iter_args, sub_indices_map, sub_insert_point)
                loop_body.add_op(scf.Yield(*resulting_iter_args))
                for_op = scf.For(lb, ext_ssa, step, iter_args, loop_body)
                Rewriter.insert_ops_at_location([for_op], insert_point)
                return for_op.res
            else:
                raise NotImplementedError(f"We do not currently support {iteration_order}")

        resulting_iter_args = _make_for_loop(iterate_op.order, iterate_op.iter_args, {}, InsertPoint.after(iterate_op))
        rewriter.replace_matched_op(ops, resulting_iter_args)

        #
        # for extent_idx, extent in reversed(list(zip(extent_idx_map, extent_loops_order))):
        #     extent_ops, (ext_ssa,) = extent_resolver.resolve(extent).output()
        #     ops.extend(extent_ops)
        #     current_loop_body = Block(
        #         arg_types=[IndexType()] + [arg.type for arg in iter_args]
        #     )
        #     current_loop_body.add_op(scf.Yield(*current_loop_body.args[1:]))
        #     loop_bodies.insert(0, current_loop_body)
        #     indices[extent_idx] = current_loop_body.args[0]
        #     iter_args = current_loop_body.args[1:1 + len(iter_args)]
        #
        #     current_loop_op = scf.For(lb, ext_ssa, step, iter_args, current_loop_body)
        #
        #     if loop_body is not None:
        #         loop_body.add_op(current_loop_op)
        #         loop_body.add_op(scf.Yield(*current_loop_op.res))
        #     else:
        #         loop_outer = current_loop_op
        #     loop_body = current_loop_body
        #
        #
        #
        # for i, extent in reversed(list(enumerate(iterate_op.extents))):
        #     extent_ops, (ext_ssa,) = extent_resolver.resolve(extent).output()
        #     ops.extend(extent_ops)
        #     loop_op = scf.For(lb, ext_ssa, step, iterate_op.iter_args, loop_body)
        #     if i > 0:
        #         block = Block(
        #             arg_types=[IndexType()] + [arg.type for arg in iterate_op.iter_args]
        #         )
        #         block.add_op(loop_op)
        #         block.add_op(scf.Yield(*loop_op.res))
        #         loop_body = block
        #         loop_bodies.append(loop_body)
        #     else:
        #         outer_loop_op = loop_op
        #
        # indices = list(reversed([body.args[0] for body in loop_bodies]))
        # assert len(indices) == len(iterate_op.extents)
        # inner_body = loop_bodies[0]
        #
        #
        # ops.append(outer_loop_op)
        # rewriter.replace_matched_op(ops, outer_loop_op.results)


@dataclass
class DLTCopyRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, copy_op: dlt.CopyOp, rewriter: PatternRewriter):
        assert all(
            isinstance(tensor.type, dlt.PtrType)
            for tensor in [copy_op.src, copy_op.dst]
        )
        src_extents = [d.extent for d in copy_op.src_dimensions]
        dst_extents = [d.extent for d in copy_op.dst_dimensions]
        assert src_extents == dst_extents
        extent_args = []
        ops = []
        for extent in src_extents:
            extent = cast(dlt.Extent, extent)
            if extent.is_init_time():
                ops.append(e_op := dlt.ExtractExtentOp(copy_op.src, extent))
                extent_args.append(e_op.res)
        iterate_op = dlt.IterateOp(
            src_extents,
            extent_args,
            [
                [[d] for d in copy_op.src_dimensions],
                [[d] for d in copy_op.dst_dimensions],
            ],
            [copy_op.src, copy_op.dst],
            [],
            dlt.NestedIterationOrderAttr.generate_for(list(range(len(src_extents))))
        )
        ops.append(iterate_op)
        body = iterate_op.body.block
        new_body_ops = []
        src = iterate_op.get_block_arg_for_tensor_arg_idx(0)
        dst = iterate_op.get_block_arg_for_tensor_arg_idx(1)
        new_body_ops.append(load := dlt.GetOp(src, copy_op.copy_type))
        new_body_ops.append(store := dlt.SetOp(dst, copy_op.copy_type, load))
        body.insert_ops_before(new_body_ops, body.last_op)
        rewriter.replace_matched_op(ops)


@dataclass
class DLTExtractExtentRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, extract_op: dlt.ExtractExtentOp, rewriter: PatternRewriter
    ):
        assert isinstance(extract_op.tree.type, dlt.PtrType)
        dlt_ptr: dlt.PtrType = extract_op.tree.type
        # dlt_ptr = cast(dlt.PtrType, dlt_ptr)
        ops = []

        llvm_type = Semantic_Map.get_data_type_from_dlt_ptr(dlt_ptr)
        ops.append(
            cast_op := builtin.UnrealizedConversionCastOp.get(
                extract_op.tree, llvm_type
            )
        )

        e_map = {
            extent: PtrCarriedExtentGetter(cast_op.outputs[0], dlt_ptr, extent=extent)
            for extent in dlt_ptr.filled_extents
        }
        extent_resolver = ExtentResolver(e_map, extract_op.get_scope())

        resolve_ops, (extent_ssa,) = extent_resolver.resolve(extract_op.extent).output()
        ops.extend(resolve_ops)
        rewriter.replace_matched_op(ops, [extent_ssa])


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
        return Semantic_Map.get_data_type_from_dlt_ptr(typ)


@dataclass
class DLTIndexTypeRewriter(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: builtin.IndexType, /) -> Attribute | None:
        return builtin.i64

@dataclass
class DLTIndexRangeTypeRewriter(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: dlt.IndexRangeType, /) -> Attribute | None:
        return llvm.LLVMStructType.from_type_list([IndexType(), IndexType()])


# def add_to_llvm_pointer(
#     ptr: SSAValue, value: SSAValue
# ) -> tuple[list[Operation], SSAValue]:
#     assert isinstance(ptr.type, llvm.LLVMPointerType)
#     ops = []
#     if isinstance(value.type, IndexType):
#         cast_ops, value = get_as_i64(value)
#         ops.extend(cast_ops)
#     ops.append(ptr_to_int_op := llvm.PtrToIntOp(ptr))
#     ops.append(add_op := arith.Addi(ptr_to_int_op.output, value))
#     ops.append(int_to_ptr_op := llvm.IntToPtrOp(add_op.result))
#     return ops, int_to_ptr_op.output


class LowerDLTPass(ModulePass):
    name = "lwer-dlt"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [DLTSelectRewriter(), DLTGetRewriter(), DLTGetSRewriter(), DLTSetRewriter()]
            )
        )
        walker.rewrite_module(op)
