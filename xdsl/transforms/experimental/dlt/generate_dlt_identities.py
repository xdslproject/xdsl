"""
Lower Data-Layout Trees into LLVM struct types (and other things)
"""

import functools
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects import func
from xdsl.dialects.builtin import ArrayAttr, IntAttr, NoneAttr, StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import SelectOp, SetAttr
from xdsl.ir import (
    Block, SSAValue, Use, Operation, Attribute,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern, TypeConversionPattern, attr_type_rewrite_pattern, GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.traits import UseDefChainTrait
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph
from xdsl.transforms.experimental.dlt.dlt_ptr_type_rewriter import PtrIdentityTypeRewriter


class _Namer:
    def __init__(self, prefix: str = "_Ident_", start: int = 0):
        self.prefix = prefix
        self.counter = start
        self.used = set()

    def get_next(self) -> StringAttr:
        ident = f"{self.prefix}{self.counter}"
        self.counter += 1
        name = StringAttr(ident)
        while name in self.used:
            ident = f"{self.prefix}{self.counter}"
            self.counter += 1
            name = StringAttr(ident)
        self.used.add(name)
        return name

    def set_used(self, name: StringAttr):
        self.used.add(name)



@dataclass
class DLTGenerateIterateOpIdentitiesRewriter(RewritePattern):

    prefix: str = "_Iter_"
    iteration_maps: dict[dlt.LayoutScopeOp, IterationMap] = field(default_factory=dict)
    namers: dict[dlt.LayoutScopeOp, _Namer] = field(default_factory=dict)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, iter_op: dlt.IterateOp, rewriter: PatternRewriter):
        scope = iter_op.get_scope()
        if scope not in self.iteration_maps:
            self.iteration_maps[scope] = IterationMap({})
        if scope not in self.namers:
            self.namers[scope] = _Namer(prefix=self.prefix)

        if iter_op.has_identity:
            self.namers[scope].set_used(iter_op.identification)
        else:
            new_name = self.namers[scope].get_next()
            iter_op.set_identity(new_name)
            rewriter.handle_operation_modification(iter_op)

        self.iteration_maps[scope].add(iter_op)



@dataclass
class DLTGeneratePtrIdentitiesRewriter(RewritePattern):

    prefix: str = "_Ident_"

    layouts: dict[dlt.LayoutScopeOp, LayoutGraph] = field(default_factory=dict)
    initial_identities: dict[dlt.LayoutScopeOp, set[StringAttr]] = field(default_factory=dict)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, scope: dlt.LayoutScopeOp, rewriter: PatternRewriter):
        scope.verify_()

        namer = _Namer(self.prefix)
        layout_graph = LayoutGraph()

        initial_identities_set: set[StringAttr] = set()

        # collect existing identities as we want to preserve them where possible
        for op in scope.walk():
            for result in [result for result in op.results if isinstance(result.type, dlt.PtrType) and result.type.has_identity]:
                initial_identities_set.add(result.type.identification)
                layout_graph.add_ssa_value(result)
            for arg in [arg for region in op.regions for block in region.blocks for arg in block.args if isinstance(arg.type, dlt.PtrType) and arg.type.has_identity]:
                initial_identities_set.add(arg.type.identification)
                layout_graph.add_ssa_value(arg)

        for name in initial_identities_set:
            namer.set_used(name)

        # give everything else that doesn't already have an identity, an identity, propagating them as we go
        for op in scope.walk():
            for result in [result for result in op.results if isinstance(result.type, dlt.PtrType)]:
                self.name_ptr_and_propagate(result, layout_graph, namer, rewriter)
            for arg in [arg for region in op.regions for block in region.blocks for arg in block.args if isinstance(arg.type, dlt.PtrType)]:
                self.name_ptr_and_propagate(arg, layout_graph, namer, rewriter)

        # add edges to graph for selections, and account for extents that are required.
        for ptr_ident, ssa_values in layout_graph.ident_count.items():
            for ssa_val, use in [(ssa_val, use) for ssa_val in ssa_values for use in ssa_val.uses]:
                if isinstance(use.operation, dlt.SelectOp):
                    select_op = cast(SelectOp, use.operation)
                    out_ptr = cast(dlt.PtrType, select_op.res.type)
                    if out_ptr.identification not in layout_graph.ident_count:
                        raise ValueError()
                    layout_graph.add_edge(ptr_ident, select_op.members, select_op.dimensions, out_ptr.identification, None)
                elif isinstance(use.operation, dlt.IterateOp):
                    iterate_op = cast(dlt.IterateOp, use.operation)
                    block_arg, dimses = iterate_op.get_block_arg_and_dims_for_input_arg(use)
                    if dimses is not None:
                        dim_attrs = {dim for dims in dimses for dim in dims}
                        out_ptr = cast(dlt.PtrType, block_arg.type)
                        layout_graph.add_edge(ptr_ident, [], dim_attrs, out_ptr.identification, iterate_op.identification)
                elif isinstance(use.operation, dlt.ExtractExtentOp):
                    extract_op = cast(dlt.ExtractExtentOp, use.operation)
                    if isinstance(extract_op.extent, dlt.InitDefinedExtentAttr):
                        layout_graph.add_extent_constraint(ptr_ident, extract_op.extent)
                else:
                    # used somewhere else? but we don't really care as select/Iterate is the only thing that constrains changes to the the ptr type
                    pass

        self.layouts[scope] = layout_graph
        self.initial_identities[scope] = initial_identities_set

        # Update all function_types as these are broken by all the identifying steps
        for op in scope.walk():
            if isinstance(op, func.FuncOp):
                op = cast(func.FuncOp, op)
                op.update_function_type()

    def name_ptr_and_propagate(self, result: SSAValue,
                               layout_graph: LayoutGraph,
                               namer: _Namer,
                               rewriter: PatternRewriter):
        assert isinstance(result.type, dlt.PtrType)
        current_ptr = cast(dlt.PtrType, result.type)
        if not current_ptr.has_identity:
            ident = namer.get_next()
            new_ptr = current_ptr.with_identification(ident)
            result.type = new_ptr
            modified_op = result.owner if isinstance(result.owner, Operation) else result.owner.parent_op()
            rewriter.handle_operation_modification(modified_op)
            layout_graph.add_ssa_value(result)
            self.propagate_ident(result, current_ptr, ident, layout_graph, rewriter)
        else:
            layout_graph.add_ssa_value(result)

    def propagate_ident(self, result: SSAValue,
                        old_ptr: dlt.PtrType,
                        identity: StringAttr,
                        layout_graph: LayoutGraph,
                        rewriter: PatternRewriter):
        assert result.type == old_ptr.with_identification(identity)
        for use in result.uses:
            ssa_values = UseDefChainTrait.get_defs_following_from_operand(use)
            for ssa in ssa_values:
                this_ptr = typing.cast(dlt.PtrType, ssa.type)
                assert isinstance(this_ptr, dlt.PtrType)
                if this_ptr.has_identity:
                    layout_graph.add_equality_edge(identity, this_ptr.identification)
                else:
                    new_ptr = old_ptr.with_identification(identity)
                    ssa.type = new_ptr
                    modified_op = ssa.owner if isinstance(ssa.owner, Operation) else ssa.owner.parent_op()
                    rewriter.handle_operation_modification(modified_op)
                    layout_graph.add_ssa_value(ssa)
                    self.propagate_ident(ssa, old_ptr, identity, layout_graph, rewriter)


@dataclass
class DLTSimplifyPtrIdentitiesRewriter(RewritePattern):
    layouts: dict[dlt.LayoutScopeOp, LayoutGraph] = field(default_factory=dict)
    initial_identities: dict[dlt.LayoutScopeOp, set[StringAttr]] = field(default_factory=dict)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, scope: dlt.LayoutScopeOp, rewriter: PatternRewriter):
        if scope in self.layouts:
            layout_graph = self.layouts[scope]
            if scope in self.initial_identities:
                initial_identities_set = self.initial_identities[scope]
            else:
                initial_identities_set = set()

            # organise identities into groups that are identical (have equality edges)
            identical_groups = layout_graph.get_equality_groups()

            # replace identities in groups to simplify the graph
            type_rewriters = []
            for group in identical_groups:
                if len(group) > 1:
                    leaders = [ident for ident in group if ident in initial_identities_set]
                    leader = leaders[0] if len(leaders) else list(group)[0]

                    assert len(layout_graph.ident_count[leader]) > 0
                    new_ptr = list(layout_graph.ident_count[leader])[0].type
                    type_rewriters.append(PtrIdentityTypeRewriter(new_ptr, group))

            type_rewriter = PatternRewriteWalker(GreedyRewritePatternApplier(type_rewriters), listener=rewriter)
            type_rewriter.rewrite_op(scope)


