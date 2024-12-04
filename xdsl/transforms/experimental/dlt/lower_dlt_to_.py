import typing
from abc import ABC
from typing import assert_type, cast

from xdsl.dialects import affine, arith, builtin, llvm, printf, scf
from xdsl.dialects.builtin import (
    AffineMapAttr, AnyFloat,
    ArrayAttr, DenseArrayBase,
    Float32Type, IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    UnrealizedConversionCastOp,
    i64,
)
from xdsl.dialects.experimental import dlt
from xdsl.ir import Attribute, Block, BlockArgument, MLContext, Operation, Region, SSAValue
from xdsl.ir.affine import AffineMap
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
    Callback,
    ExtentResolver,
    IndexGetter,
    LoopCallback, NoCallback,
    PtrCarriedExtentGetter,
    PtrCarriedIndexGetter,
    SSAExtentGetter,
    SemanticsMapper, load_all_semantics,
    ValueMapInitialiser,
)


def get_as_i64(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(value.type, IndexType)
    if isinstance(value.type, IntegerType):
        assert (
            value.type.width.data <= i64.width.data
        ), f"Expected {i64.width.data} got {value.type.width.data}"
    return [op := arith.IndexCastOp(value, i64)], op.result
    # return [op := UnrealizedConversionCastOp.get([value], [i64])], op.outputs[0]

class DLTRewritePattern(RewritePattern, ABC):

    def __init__(self, semantics: SemanticsMapper):
        self.semantics = semantics


class DLTSelectRewriter(DLTRewritePattern):

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

        llvm_in_ptr_type = self.semantics.get_data_type_from_dlt_ptr(input_type)
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

        select_ops, select_result = self.semantics.get_select_for(
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
            gen_ops, ptr_struct_result = self.semantics.generate_ptr_struct(
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


class DLTGetRewriter(DLTRewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_op: dlt.GetOp, rewriter: PatternRewriter):
        assert isinstance(get_op.tree.type, dlt.PtrType)
        input_type = cast(dlt.PtrType, get_op.tree.type)
        assert isinstance(input_type, dlt.PtrType)
        get_type = assert_type(get_op.get_type, dlt.AcceptedTypes)
        assert get_op.res.type == get_op.get_type
        assert isinstance(get_type, dlt.AcceptedTypes)

        ops = []
        llvm_in_ptr_type = self.semantics.get_data_type_from_dlt_ptr(input_type)
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

        get_ops, get_res, get_found = self.semantics.get_getter_for(
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
                get_op.get_scope(),
            ),
            llvm_data_ptr,
        )
        ops.extend(get_ops)
        if get_found is not True:
            zero_ops, zero_val = (
                layout_llvm_semantics._get_packed_zero_for_accepted_type(get_type)
            )
            ops.extend(zero_ops)
            if isinstance(get_found, SSAValue):
                ops.append(select_op := arith.Select(get_found, get_res, zero_val))
                get_res = select_op.result
            else:
                get_res = zero_val

        rewriter.replace_matched_op(ops, [get_res])


class DLTGetSRewriter(DLTRewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_op: dlt.GetSOp, rewriter: PatternRewriter):
        assert isinstance(get_op.tree.type, dlt.PtrType)
        input_type = cast(dlt.PtrType, get_op.tree.type)
        assert isinstance(input_type, dlt.PtrType)
        get_type = assert_type(get_op.get_type, dlt.AcceptedTypes)
        assert get_op.res.type == get_op.get_type
        assert isinstance(get_type, dlt.AcceptedTypes)

        ops = []
        llvm_in_ptr_type = self.semantics.get_data_type_from_dlt_ptr(input_type)
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

        get_ops, get_res, get_found = self.semantics.get_getter_for(
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
                get_op.get_scope(),
            ),
            llvm_data_ptr,
        )
        ops.extend(get_ops)

        bool_ops, get_found = layout_llvm_semantics._make_bool_ssa(get_found)
        ops.extend(bool_ops)

        zero_ops, zero_val = layout_llvm_semantics._get_packed_zero_for_accepted_type(
            get_type
        )
        ops.extend(zero_ops)

        ops.append(select_op := arith.Select(get_found, get_res, zero_val))
        gets_res = select_op.result
        rewriter.replace_matched_op(ops, [gets_res, get_found])


class DLTSetRewriter(DLTRewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, set_op: dlt.SetOp, rewriter: PatternRewriter):
        assert isinstance(set_op.tree.type, dlt.PtrType)
        input_type = cast(dlt.PtrType, set_op.tree.type)
        assert isinstance(input_type, dlt.PtrType)
        set_type = assert_type(set_op.set_type, dlt.AcceptedTypes)
        assert set_op.value.type == set_op.set_type
        assert isinstance(set_type, dlt.AcceptedTypes)

        ops = []
        llvm_in_ptr_type = self.semantics.get_data_type_from_dlt_ptr(input_type)
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

        set_ops = self.semantics.get_setter_for(
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
                set_op.get_scope(),
            ),
            llvm_data_ptr,
        )
        ops.extend(set_ops)

        rewriter.replace_matched_op(ops, [])


class DLTAllocRewriter(DLTRewritePattern):

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
            self.semantics.get_size(ptr_type.layout, extent_resolver).sum().output()
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
        if self.semantics.print_memory_calls:
            ops.append(printf.PrintFormatOp("# called malloc({}) -> {}", alloc_bytes, malloc.returned))
        buffer = malloc.returned

        gen_ops, ptr_struct = self.semantics.generate_ptr_struct(
            ptr_type, buffer, {}, extent_resolver
        )
        ops.extend(gen_ops)

        init_values_map = {
            cast(dlt.PtrType, init_arg.type).contents_type: init_arg
            for init_arg in alloc_op.initialValues
        }
        value_map_initialiser = ValueMapInitialiser(
            self.semantics, extent_resolver, init_values_map
        )

        # init_ops = init_layout(
        #     ptr_type.layout, extent_resolver, buffer, init_values_map
        # )
        init_ops, callback_ret = self.semantics.init_layout(
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


class DLTDeallocRewriter(DLTRewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, dealloc_op: dlt.DeallocOp, rewriter: PatternRewriter):
        assert isinstance(dealloc_op.tree.type, dlt.PtrType)
        ptr_type = cast(dlt.PtrType, dealloc_op.tree.type)

        ops = []

        llvm_in_ptr_type = self.semantics.get_data_type_from_dlt_ptr(ptr_type)
        cast_input_op = UnrealizedConversionCastOp.get(
            dealloc_op.tree, llvm_in_ptr_type
        )
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

        dealloc_ops = self.semantics.dealloc_layout(
            ptr_type.layout,
            extent_resolver,
            llvm_data_ptr,
        )

        ops.extend(dealloc_ops)

        ops.append(free := llvm.CallOp("free", llvm_data_ptr, return_type=None))
        if self.semantics.print_memory_calls:
            ops.append(printf.PrintFormatOp("# called free({})", llvm_data_ptr))

        rewriter.replace_matched_op(ops, [])


class DLTIterateRewriter(DLTRewritePattern):

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
        full_extent_map = dict(extent_map)

        tensor_map = []

        for tensor_arg, tensor_dims in zip(iterate_op.tensors, iterate_op.dimensions):
            tensor_dims = cast(
                builtin.ArrayAttr[dlt.SetAttr[dlt.DimensionAttr]], tensor_dims
            )
            ops.append(
                cast_op := builtin.UnrealizedConversionCastOp.get(
                    tensor_arg, self.semantics.get_data_type_from_dlt_ptr(tensor_arg.type)
                )
            )
            llvm_tensor_arg = cast_op.outputs[0]
            e_map = {
                extent: PtrCarriedExtentGetter(
                    llvm_tensor_arg, tensor_arg.type, extent=extent
                )
                for extent in tensor_arg.type.filled_extents
            }
            full_extent_map |= e_map
            tensor_arg_extent_resolver = ExtentResolver(e_map, iterate_op.get_scope())

            tensor_map_map = {}
            for extent_index, (extent, ext_dims) in enumerate(
                zip(iterate_op.extents, tensor_dims)
            ):
                ext_dims = cast(dlt.SetAttr[dlt.DimensionAttr], ext_dims)
                if len(ext_dims) == 0:
                    pass  # this extent isn't used in this tensor - so it need not match, or exist
                elif extent.is_static():
                    assert all(dim.extent == extent for dim in ext_dims)
                elif extent.is_scope_time():
                    assert all(dim.extent == extent for dim in ext_dims)
                else:
                    # extent is used but cannot be compared statically, so we check bounds before looping.
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
                for ext_dim in ext_dims:
                    tensor_map_map[ext_dim] = extent_index

            tensor_map.append((tensor_arg, tensor_map_map))

        full_extent_resolver = ExtentResolver(full_extent_map, iterate_op.get_scope())

        resulting_iter_args = self._make_for_loop(
            iterate_op,
            full_extent_resolver,
            tensor_map,
            iterate_op.order,
            list(iterate_op.iter_args),
            {},
            InsertPoint.after(iterate_op),
            ops,
        )
        rewriter.replace_matched_op(ops, resulting_iter_args)

    def _make_for_loop(
        self,
        iterate_op: dlt.IterateOp,
        extent_resolver: ExtentResolver,
        tensor_map: list[tuple[SSAValue, dict[dlt.DimensionAttr, int]]],
        iteration_order: dlt.IterationOrder,
        iter_args: list[SSAValue],
        indices_map: dict[int, BlockArgument],
        insert_point: InsertPoint,
        pre_ops: list[Operation],
    ) -> list[SSAValue]:
        if isinstance(iteration_order, dlt.BodyIterationOrderAttr):
            resulting_iter_args = self._make_inner_body(
                iterate_op,
                extent_resolver,
                tensor_map,
                indices_map,
                iter_args,
                insert_point,
                pre_ops,
            )
            return resulting_iter_args
        elif isinstance(iteration_order, dlt.NestedIterationOrderAttr):
            pre_ops.append(lb := arith.Constant(IntegerAttr(0, IndexType())))
            pre_ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))
            extent_idx = iteration_order.extent_index.data
            extent = iterate_op.extents.data[extent_idx]
            extent_ops, (ext_ssa,) = extent_resolver.resolve(extent).output()
            pre_ops.extend(extent_ops)
            loop_body = Block(arg_types=[IndexType()] + [arg.type for arg in iter_args])
            sub_iter_args = list(loop_body.args[1:])
            sub_indices_map = indices_map | {extent_idx: loop_body.args[0]}
            sub_insert_point = InsertPoint.at_end(loop_body)
            new_tensor_map = []
            for tensor_ssa, t_map in tensor_map:
                dims = []
                values = []
                for dim, e_idx in t_map.items():
                    if e_idx == extent_idx:
                        dims.append(dim)
                        values.append(sub_indices_map[e_idx])
                new_t_map = {d: i for d, i in t_map.items() if d not in dims}
                select_op = dlt.SelectOp(tensor_ssa, [], dims, values)
                loop_body.add_op(select_op)
                new_tensor_map.append((select_op.res, new_t_map))
            resulting_iter_args = self._make_for_loop(
                iterate_op,
                extent_resolver,
                new_tensor_map,
                iteration_order.child,
                sub_iter_args,
                sub_indices_map,
                sub_insert_point,
                pre_ops,
            )
            # loop_body.add_op(scf.Yield(*resulting_iter_args))
            loop_body.add_op(affine.Yield.get(*resulting_iter_args))
            result_types = [v.type for v in resulting_iter_args]
            # for_op = scf.For(lb, ext_ssa, step, iter_args, loop_body)
            affine_for_op = affine.For.from_region([lb], [ext_ssa], iter_args, result_types, AffineMapAttr(AffineMap.identity(1)), AffineMapAttr(AffineMap.identity(1)), Region(loop_body))
            Rewriter.insert_ops_at_location([affine_for_op], insert_point)
            return list(affine_for_op.res)
        elif isinstance(iteration_order, dlt.NonZeroIterationOrderAttr):
            ops = []
            sparse_tensor_idx = iteration_order.tensor_index.data
            sparse_tensor_ssa, dims = tensor_map[sparse_tensor_idx]
            sparse_tensor_ptr_type = typing.cast(dlt.PtrType, sparse_tensor_ssa.type)
            inner_ptr_type = iterate_op.get_block_arg_for_tensor_arg_idx(
                sparse_tensor_idx
            ).type
            inner_ptr_type = typing.cast(dlt.PtrType, inner_ptr_type)
            sparse_tensor_llvm_type = self.semantics.get_data_type_from_dlt_ptr(
                sparse_tensor_ptr_type
            )
            cast_op = UnrealizedConversionCastOp.get(
                sparse_tensor_ssa, sparse_tensor_llvm_type
            )
            ops.append(cast_op)
            sparse_tensor_llvm = cast_op.outputs[0]

            get_data_ptr_op = llvm.ExtractValueOp(
                DenseArrayBase.from_list(i64, [0]),
                sparse_tensor_llvm,
                llvm.LLVMPointerType.opaque(),
            )
            ops.append(get_data_ptr_op)
            sparse_tensor_llvm_data_ptr = get_data_ptr_op.res

            filled_members = set(sparse_tensor_ptr_type.filled_members)
            filled_dims = {
                dim: PtrCarriedIndexGetter(
                    sparse_tensor_llvm, sparse_tensor_ptr_type, dim=dim
                )
                for dim in sparse_tensor_ptr_type.filled_dimensions
            }
            extent_idxs_to_loop = [e.data for e in iteration_order.extent_indices]
            extents_to_loop = [iterate_op.extents.data[e] for e in extent_idxs_to_loop]
            dims_to_loop = {
                d
                for e in extent_idxs_to_loop
                for d in iterate_op.dimensions.data[sparse_tensor_idx].data[e]
            }

            callback = _IterMakeLoopCallback(iterate_op,
                                             extent_resolver,
                                             tensor_map,
                                             iteration_order,
                                             iter_args,
                                             indices_map,
                                             dims,
                                             pre_ops,
                                             )

            Rewriter.insert_ops_at_location(ops, insert_point)

            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                sparse_tensor_ptr_type.layout,
                inner_ptr_type.layout,
                extent_resolver,
                sparse_tensor_llvm_data_ptr,
                callback,
                callback.initial_iter_args(),
                filled_members,
                filled_dims,
                dims_to_loop,
            )
            Rewriter.insert_ops_at_location(child_ops, insert_point)
            return iter_args_out
        else:
            raise NotImplementedError(f"We do not currently support {iteration_order}")

    def _make_inner_body(
            self,
            iterate_op: dlt.IterateOp,
        extent_resolver: ExtentResolver,
        tensor_map: list[tuple[SSAValue, dict[dlt.DimensionAttr, int]]],
        indices_map: dict[int, BlockArgument],
        iter_args: list[SSAValue],
        insert_point: InsertPoint,
        pre_ops: list[Operation],
    ) -> list[SSAValue]:
        _inner_body_ops = []
        # debug_string = f"Iter: '{iterate_op.identification.data}' : extents: "
        # args = []
        # for ext_id, val in indices_map.items():
        #     debug_string += f"{ext_id}: {{}}, "
        #     args.append(val)
        # debug_string += "tensors: "
        # for t_i, (t_ssa, t_map) in enumerate(tensor_map):
        #     debug_string += f"{t_i}:'{t_ssa.type.identification.data}':{{}} "
        #     extract_ops, data_ptr, ptr_dim_map, ptr_extent_map = Semantic_Map.extract_from_ptr_struct(
        #         t_ssa.type, t_ssa)
        #     _inner_body_ops.extend(extract_ops)
        #     ptr_int = llvm.PtrToIntOp(data_ptr)
        #     _inner_body_ops.append(ptr_int)
        #     args.append(ptr_int.output)
        #     debug_string += "["
        #     for dim, val in ptr_dim_map.items():
        #         get_ops, (ssa,) = val.get().output()
        #         _inner_body_ops.extend(get_ops)
        #         debug_string += f"{dim.dimensionName.data}:{{}}"
        #         args.append(ssa)
        #     debug_string += "]"
        #
        #
        # debug_string += "iter_args: "
        # for i, ssa in enumerate(iter_args):
        #     if isinstance(ssa.type, IntegerType | Float32Type | IndexType):
        #         debug_string += f"{i}: {{}}"
        #         args.append(ssa)
        #
        # print_op = printf.PrintFormatOp(debug_string, *args)
        # _inner_body_ops.append(print_op)

        selected_tensors = []
        block_arg_tensor_types = [
            arg.type for arg in iterate_op.get_block_args_for_tensor_args()
        ]
        indices = [indices_map[i] for i in range(len(iterate_op.extents))]

        for (tensor_ssa, tensor_dim_map), tensor_type in zip(
            tensor_map, block_arg_tensor_types
        ):
            dims = []
            values = []
            for dim, extent_index in tensor_dim_map.items():
                dims.append(dim)
                values.append(indices_map[extent_index])
            select = dlt.SelectOp(tensor_ssa, [], dims, values, tensor_type)
            selected_tensors.append(select.res)
            _inner_body_ops.append(select)

        iterate_op_body_arg_vals = indices + selected_tensors + iter_args
        dlt_yield_op = iterate_op.get_yield_op()
        yielded = dlt_yield_op.arguments
        dlt_yield_op.detach()
        dlt_yield_op.erase()
        Rewriter.insert_ops_at_location(_inner_body_ops, insert_point)
        Rewriter.inline_block_at_location(
            iterate_op.body.block, insert_point, iterate_op_body_arg_vals
        )

        return yielded


class _IterMakeLoopCallback(LoopCallback):

    def __init__(
            self,
            iterate_rewriter: DLTIterateRewriter,
            iterate_op: dlt.IterateOp,
            extent_resolver: ExtentResolver,
            tensor_map: list[tuple[SSAValue, dict[dlt.DimensionAttr, int]]],
            current_iter_order: dlt.NonZeroIterationOrderAttr,
            iter_args: list[SSAValue],
            indices_map: dict[int, BlockArgument],
            dim_to_extent_idx_map: dict[dlt.DimensionAttr, int],
            pre_ops: list[Operation],

    ):
        super().__init__(iter_args)
        self.iterate_rewriter = iterate_rewriter
        self.iterate_op = iterate_op
        self.extent_resolver = extent_resolver
        self.tensor_map = tensor_map
        self.current_iter_order = current_iter_order
        self.indices_map = indices_map
        self.dim_to_extent_idx_map = dim_to_extent_idx_map
        self.pre_ops = pre_ops

    def body(self,
             terminal_layout: dlt.Layout,
             members: set[dlt.MemberAttr],
             dim_map: dict[dlt.DimensionAttr, IndexGetter],
             extent_resolver: ExtentResolver,
             ptr: SSAValue,
             iter_args: list[SSAValue],
             ) -> tuple[list[Operation], list[SSAValue]]:
        tensor_idx = self.current_iter_order.tensor_index.data
        tensor_ptr, dims = self.tensor_map[tensor_idx]
        extent_indices = {dims[d]: dim_map[d] for d in dims}
        assert all(e.data in extent_indices for e in self.current_iter_order.extent_indices)
        ops = []
        new_tensor_map = []
        for t_idx, (tensor_ssa, t_map) in enumerate(self.tensor_map):
            dims = []
            values = []
            for dim, e_idx in t_map.items():
                if e_idx in extent_indices:
                    dims.append(dim)
                    val_op, (val,) = extent_indices[e_idx].get().output()
                    ops.extend(val_op)
                    values.append(val)
            new_t_map = {d: i for d, i in t_map.items() if d not in dims}
            if t_idx == tensor_idx:

                tensor_ptr_type = typing.cast(dlt.PtrType, tensor_ssa.type)
                extract_ops, data_ptr, ptr_dim_map, ptr_extent_map = self.iterate_rewriter.semantics.extract_from_ptr_struct(
                    tensor_ptr_type, tensor_ssa)
                ops.extend(extract_ops)

                new_members = members - set(tensor_ptr_type.filled_members)
                new_dimensions = set(dim_map.keys()) - set(ptr_dim_map.keys())
                new_contents_type = tensor_ptr_type.contents_type.select_members(new_members).select_dimensions(new_dimensions)
                selection_needed = terminal_layout.contents_type.has_selectable_type(new_contents_type)
                assert len(selection_needed) == 1
                members_needed, dimensions_needed = selection_needed.pop()
                extents_needed = terminal_layout.get_all_init_base_extents()
                terminal_layout.contents_type.has_selectable_type(tensor_ptr_type.contents_type.select_members(new_members).select_dimensions(new_dimensions))
                new_tensor_ptr_type = dlt.PtrType(
                    new_contents_type,
                    terminal_layout,
                    members_needed,
                    ArrayAttr(dimensions_needed),
                    ArrayAttr(extents_needed),
                )
                gen_ops, ptr_struct_result = self.iterate_rewriter.semantics.generate_ptr_struct(
                    new_tensor_ptr_type, ptr, ptr_dim_map | dim_map, self.extent_resolver.with_new(ptr_extent_map)
                )
                ops.extend(gen_ops)
                cast_output_op = UnrealizedConversionCastOp.get(ptr_struct_result, new_tensor_ptr_type)
                ops.append(cast_output_op)
                new_tensor_map.append((cast_output_op.outputs[0], new_t_map))
            else:
                select_op = dlt.SelectOp(tensor_ssa, [], dims, values)
                ops.append(select_op)
                new_tensor_map.append((select_op.res, new_t_map))

        sub_indices_map = dict(self.indices_map)
        for i, getter in extent_indices.items():
            getter_ops, (value,) = getter.get().output()
            ops.extend(getter_ops)
            sub_indices_map[i] = value

        loop_block = Block()
        resulting_iter_args = self.iterate_rewriter._make_for_loop(
            self.iterate_op,
            self.extent_resolver,
            new_tensor_map,
            self.current_iter_order.child,
            iter_args,
            sub_indices_map,
            InsertPoint.at_end(loop_block),
            self.pre_ops,
        )
        while len(loop_block.ops) > 0:
            op = loop_block.detach_op(loop_block.first_op)
            ops.append(op)

        return ops, resulting_iter_args

    def callback(
        self,
        terminal_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, IndexGetter],
        dims_left_to_loop: set[dlt.DimensionAttr],
        extent_resolver: ExtentResolver,
        ptr: SSAValue,
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert (set(dim_map.keys()) & dims_left_to_loop) == set()
        if len(dims_left_to_loop) == 0:
            return self.body(terminal_layout, members, dim_map, extent_resolver, ptr, iter_args)
        dims_left_to_loop = sorted(list(dims_left_to_loop), key=(lambda d:d.dimensionName.data))
        dim_to_loop = dims_left_to_loop.pop()
        ops = []
        ops.append(lb := arith.Constant(IntegerAttr(0, IndexType())))
        ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))

        extent = dim_to_loop.extent
        extent_ops, (ext_ssa,) = extent_resolver.resolve(extent).output()
        ops.extend(extent_ops)

        loop_body = Block(arg_types=[IndexType()] + [arg.type for arg in iter_args])
        sub_index = loop_body.args[0]

        new_dim_map = dim_map | {dim_to_loop : ArgIndexGetter(sub_index)}
        loop_iter_args = list(loop_body.args[1:])

        loop_ops, iter_results = self.callback(
            terminal_layout,
            members,
            new_dim_map,
            set(dims_left_to_loop),
            extent_resolver,
            ptr,
            loop_iter_args,
        )
        loop_body.add_ops(loop_ops)
        loop_body.add_op(scf.Yield(*iter_results))
        for_op = scf.For(lb, ext_ssa, step, iter_args, loop_body)
        ops.append(for_op)
        return ops, list(for_op.res)


class DLTCopyRewriter(DLTRewritePattern):

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
            dlt.NestedIterationOrderAttr.generate_for(list(range(len(src_extents)))),
        )
        ops.append(iterate_op)
        body = iterate_op.body.block
        new_body_ops = []
        src = iterate_op.get_block_arg_for_tensor_arg_idx(0)
        dst = iterate_op.get_block_arg_for_tensor_arg_idx(1)
        new_body_ops.append(load := dlt.GetOp(src, copy_op.copy_type))
        new_body_ops.append(store := dlt.SetOp(dst, copy_op.copy_type, load.res))
        body.insert_ops_before(new_body_ops, body.last_op)
        rewriter.replace_matched_op(ops)


class DLTExtractExtentRewriter(DLTRewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, extract_op: dlt.ExtractExtentOp, rewriter: PatternRewriter
    ):
        assert isinstance(extract_op.tree.type, dlt.PtrType)
        dlt_ptr: dlt.PtrType = extract_op.tree.type
        # dlt_ptr = cast(dlt.PtrType, dlt_ptr)
        ops = []

        llvm_type = self.semantics.get_data_type_from_dlt_ptr(dlt_ptr)
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


class DLTScopeRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, scope_op: dlt.LayoutScopeOp, rewriter: PatternRewriter):
        rewriter.inline_block_before_matched_op(scope_op.body.block)
        rewriter.erase_matched_op()


class DLTPtrTypeRewriter(TypeConversionPattern):


    def __init__(self, semantics: SemanticsMapper, recursive: bool = False, ops: tuple[type[Operation]]|None = None):
        self.semantics = semantics
        super().__init__(recursive, ops)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: dlt.PtrType, /) -> Attribute | None:
        return self.semantics.get_data_type_from_dlt_ptr(typ)


class DLTIndexTypeRewriter(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: builtin.IndexType, /) -> Attribute | None:
        return builtin.i64


class DLTIndexRangeTypeRewriter(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: dlt.IndexRangeType, /) -> Attribute | None:
        return llvm.LLVMStructType.from_type_list([i64, i64])



# class LowerDLTPass(ModulePass):
#     name = "lwer-dlt"
#
#     def apply(self, ctx: MLContext, op: ModuleOp) -> None:
#         walker = PatternRewriteWalker(
#             GreedyRewritePatternApplier(
#                 [
#                     DLTSelectRewriter(),
#                     DLTGetRewriter(),
#                     DLTGetSRewriter(),
#                     DLTSetRewriter(),
#                 ]
#             )
#         )
#         walker.rewrite_module(op)
