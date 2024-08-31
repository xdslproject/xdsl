import typing
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, NoneAttr
from xdsl.dialects.experimental import dlt
from xdsl.ir import BlockArgument
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class DLTIterateOptimiserRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, sel_op: dlt.SelectOp, rewriter: PatternRewriter):
        parents = []
        current = sel_op.parent_op()
        while isinstance(current, dlt.IterateOp):
            current = typing.cast(dlt.IterateOp, current)
            if sel_op.tree in current.get_block_args_for_tensor_args():
                break
            if current.is_ancestor(sel_op.tree.owner):
                break
            parents.append(current)
            current = current.parent_op()
        if len(parents) < 1:
            return

        dim_mapping_per_parent = {}
        dims_to_remove_per_parent = {}
        dims_to_remove = []
        for dim, value in zip(sel_op.dimensions, sel_op.values):
            if (
                isinstance(value, BlockArgument)
                and value.owner.parent_op() in parents
                and value
                in [a for p in parents for a in p.get_block_args_for_extent_args()]
            ):
                # found a select that uses an extent but isn't part of a parent Iterate op
                dims_to_remove.append(dim)

                parent_idx = [
                    (p_i, p.get_block_args_for_extent_args().index(value))
                    for p_i, p in enumerate(parents)
                    if value in p.get_block_args_for_extent_args()
                ]
                assert len(parent_idx) == 1
                parent_idx, extent_idx = parent_idx[0]
                dim_mapping_per_parent.setdefault(parent_idx, {}).setdefault(
                    extent_idx, set()
                ).add(dim)
                dims_to_remove_per_parent.setdefault(parent_idx, set()).add(dim)

        if len(dims_to_remove) < 1:
            return  # No optimisations to be done here

        new_trees = [sel_op.tree]
        for p_i, parent in reversed(list(enumerate(parents))):
            new_arg_type = dlt.SelectOp.calculateResultType(
                new_trees[0].type, [], dims_to_remove_per_parent.get(p_i, [])
            )
            new_tree = parent.body.block.insert_arg(
                new_arg_type, len(parent.extents) + len(parent.tensors)
            )
            new_trees.insert(0, new_tree)

        dim_val_pairs = [
            (d, e)
            for d, e in zip(sel_op.dimensions, sel_op.values)
            if d not in dims_to_remove
        ]
        new_select_dimensions = [d for d, e in dim_val_pairs]
        new_values = [e for d, e in dim_val_pairs]
        new_select = dlt.SelectOp(
            new_trees[0],
            sel_op.members,
            new_select_dimensions,
            new_values,
            sel_op.res.type,
        )
        zero_group = None
        if "select_index_group" in sel_op.attributes:
            zero_group = sel_op.attributes["select_index_group"]
            new_select.attributes["select_index_group"] = zero_group

        rewriter.replace_matched_op([new_select], [new_select.res])

        for p_i, parent in enumerate(parents):
            new_iterate_body = parent.body.detach_block(parent.body.block)
            tensor_dimensions = []
            for extent_idx, e in enumerate(parent.extents):
                dims = dim_mapping_per_parent.get(p_i, {}).get(extent_idx, set())
                tensor_dimensions.append(dims)
            new_iter_dimensions = [
                [[dss for dss in ds] for ds in d] for d in parent.dimensions
            ] + [tensor_dimensions]

            new_iter_tensors = (*parent.tensors, new_trees[p_i + 1])

            if "tensor_select_index_groups" in parent.attributes:
                if zero_group is None:
                    zero_group = NoneAttr()
                parent_tensor_zero_groups = typing.cast(ArrayAttr, parent.attributes["tensor_select_index_groups"])
                assert isinstance(parent_tensor_zero_groups, ArrayAttr)
                assert len(parent_tensor_zero_groups) == len(parent.tensors)
                parent_tensor_zero_groups = list(parent_tensor_zero_groups)
            else:
                parent_tensor_zero_groups = [NoneAttr()]*len(parent.tensors)


            new_iterate_op = dlt.IterateOp(
                list(parent.extents),
                parent.extent_args,
                new_iter_dimensions,
                new_iter_tensors,
                parent.iter_args,
                parent.order,
                parent.identification,
                new_iterate_body,
            )
            if "zeroable" in parent.attributes:
                new_iterate_op.attributes["zeroable"] = parent.attributes["zeroable"]
            if zero_group is not None:
                new_tensor_zero_groups = parent_tensor_zero_groups + [zero_group]
                new_iterate_op.attributes["tensor_select_index_groups"] = ArrayAttr(new_tensor_zero_groups)

            rewriter.replace_op(parent, [new_iterate_op], new_iterate_op.results)
