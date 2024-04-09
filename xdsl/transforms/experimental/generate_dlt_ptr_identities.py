"""
Lower Data-Layout Trees into LLVM struct types (and other things)
"""

import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects import func
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import SelectOp
from xdsl.ir import (
    SSAValue, Use, Operation,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class DLTGeneratePtrIdentitiesRewriter(RewritePattern):

    prefix: str = "_Ident_"
    identity_num = 0
    ident_count: defaultdict[StringAttr, set[SSAValue]] = field(default_factory=lambda: defaultdict(set))
    initial_identities: list[StringAttr] = field(default_factory=lambda: list())
    func_calls: dict[StringAttr, tuple[func.FuncOp, set[func.Call]]] = field(default_factory=lambda: dict())
    ident_replacements: defaultdict[StringAttr, set[StringAttr]] = field(default_factory=lambda: defaultdict(set))
    graph_edges: dict[
        StringAttr, set[tuple[dlt.SetAttr[dlt.MemberAttr], dlt.SetAttr[dlt.DimensionAttr], StringAttr]]] = field(
        default_factory=lambda: dict())
    required_extents: dict[
        StringAttr, set[dlt.InitDefinedExtentAttr]] = field(
        default_factory=lambda: dict())

    def _next_ident(self) -> StringAttr:
        ident = f"{self.prefix}{self.identity_num}"
        self.identity_num += 1
        return StringAttr(ident)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, scope: dlt.LayoutScopeOp, rewriter: PatternRewriter):
        scope.verify_()
        for op in scope.walk():
            if isinstance(op, func.FuncOp):
                func_op = cast(func.FuncOp, op)
                assert func_op.sym_name not in self.func_calls
                self.func_calls[func_op.sym_name] = (func_op, set())
            for result in [result for result in op.results if isinstance(result.type, dlt.PtrType) and result.type.has_identity]:
                self.initial_identities.append(result.type.identification)
                self.ident_count[result.type.identification].add(result)
            for arg in [arg for region in op.regions for block in region.blocks for arg in block.args if isinstance(arg.type, dlt.PtrType) and arg.type.has_identity]:
                self.initial_identities.append(arg.type.identification)
                self.ident_count[arg.type.identification].add(arg)

        self.initial_identities = [ident for i, ident in enumerate(self.initial_identities) if ident not in self.initial_identities[:i]]

        for op in scope.walk():
            if isinstance(op, func.Call) and (func_name := (call := cast(func.Call, op)).callee.root_reference) in self.func_calls:
                func_op, calls = self.func_calls[func_name]
                calls.add(call)

        for op in scope.walk(): ## propagate entrypoints first
            if isinstance(op, func.FuncOp):
                func_op = cast(func.FuncOp, op)
                for arg in func_op.body.block.args:
                    if isinstance(arg.type, dlt.PtrType):
                        self.name_ptr(arg)
            if isinstance(op, dlt.AllocOp):
                op = cast(dlt.AllocOp, op)
                self.name_ptr(op.res)

        for op in scope.walk(): ## give everything else that doesn't already have an identity, an identity
            for result in [result for result in op.results if isinstance(result.type, dlt.PtrType)]:
                self.name_ptr(result)
            for arg in [arg for region in op.regions for block in region.blocks for arg in block.args if isinstance(arg.type, dlt.PtrType)]:
                self.name_ptr(arg)

        if len(self.ident_replacements) > 0:
            raise NotImplementedError()


        for ptr_ident, ssa_values in self.ident_count.items():
            print(ptr_ident)
            self.graph_edges[ptr_ident] = set()
            self.required_extents[ptr_ident] = set()
            for ssa_val, use in [(ssa_val, use) for ssa_val in ssa_values for use in ssa_val.uses]:
                print(ssa_val)
                if isinstance(use.operation, dlt.SelectOp):
                    select_op = cast(SelectOp, use.operation)
                    out_ptr = cast(dlt.PtrType, select_op.res.type)
                    if out_ptr.identification not in self.ident_count:
                        raise ValueError()
                    self.graph_edges[ptr_ident].add((select_op.members, dlt.SetAttr(select_op.dimensions), out_ptr.identification))
                elif isinstance(use.operation, dlt.IterateOp):
                    iterate_op = cast(dlt.IterateOp, use.operation)
                    block_arg, dimses = iterate_op.get_block_arg_and_dims_for_input_arg(use)
                    if dimses is not None:
                        dim_attrs = dlt.SetAttr({dim for dims in dimses for dim in dims})
                        out_ptr = cast(dlt.PtrType, block_arg.type)


                elif isinstance(use.operation, dlt.ExtractExtentOp):
                    extract_op = cast(dlt.ExtractExtentOp, use.operation)
                    if isinstance(extract_op.extent, dlt.InitDefinedExtentAttr):
                        self.required_extents[ptr_ident].add(extract_op.extent)
                else:
                    # used somewhere else? but we don't really care as select is the only thing that constrains changes to the the ptr type
                    pass



        print("DONE")

    def name_ptr(self, result: SSAValue):
        assert isinstance(result.type, dlt.PtrType)
        current_ptr = cast(dlt.PtrType, result.type)
        if not current_ptr.has_identity:
            ident = self._next_ident()
            new_ptr = current_ptr.with_identification(ident)
            result.type = new_ptr
            self.ident_count[ident].add(result)
            self.propagate_ident(result, current_ptr, ident)
        # else:
        #     ident = current_ptr.identification
        #     self.propagate_ident(result, current_ptr, ident)

    def propagate_ident(self, result: SSAValue, old_ptr: dlt.PtrType, identity: StringAttr):
        assert result.type == old_ptr.with_identification(identity)
        for use in result.uses:
            self._propagate_ident(use.operation, result, use, old_ptr, identity)

    @functools.singledispatchmethod
    def _propagate_ident(self, receiving_op: Operation, operand: SSAValue, use: Use, old_ptr: dlt.PtrType, identity: StringAttr):
        for ssa in [arg for region in receiving_op.regions for block in region.blocks for arg in block.args if isinstance(arg.type, dlt.PtrType)] + [r for r in receiving_op.results if isinstance(r.type, dlt.PtrType)]:
            if ssa.type == old_ptr:
                # old_ptr needs to be replaced.
                new_ptr = old_ptr.with_identification(identity)
                ssa.type = new_ptr
                self.ident_count[identity].add(ssa)
                self.propagate_ident(ssa, old_ptr, identity)
            if ssa.type.with_identification("") == old_ptr.with_identification(""):
                #whaaa
                pass

    @_propagate_ident.register
    def _(self, call: func.Call, operand: SSAValue, use: Use, old_ptr: dlt.PtrType, identity: StringAttr):
        func_name = call.callee.root_reference
        if func_name in self.func_calls:
            func_op, calls = self.func_calls[func_name]
            func_arg = func_op.body.block.args[use.index]
            assert isinstance(func_arg.type, dlt.PtrType)
            assert func_arg.type.with_identification("") == old_ptr.with_identification("")
            if func_arg.type.has_identity:
                self.ident_replacements[func_arg.type.identification].add(identity)
                self.ident_replacements[identity].add(func_arg.type.identification)
            else:
                old_func_arg_type = func_arg.type
                func_arg.type = old_ptr.with_identification(identity)
                self.propagate_ident(func_arg, old_func_arg_type, identity)
            for call in calls:
                c_operand = call.operands[use.index]
                assert isinstance(c_operand.type, dlt.PtrType)
                assert c_operand.type.with_identification("") == old_ptr.with_identification("")
                if c_operand.type.has_identity:
                    self.ident_replacements[c_operand.type.identification].add(identity)
                    self.ident_replacements[identity].add(c_operand.type.identification)

    @_propagate_ident.register
    def _(self, iterate_op: dlt.IterateOp, operand: SSAValue, use: Use, old_ptr: dlt.PtrType, identity: StringAttr):
        block_arg, dims = iterate_op.get_block_arg_and_dims_for_input_arg(use)
        if dims is None:
            assert block_arg.type == old_ptr
            # old_ptr needs to be replaced.
            new_ptr = old_ptr.with_identification(identity)
            block_arg.type = new_ptr
            self.ident_count[identity].add(block_arg)
            self.propagate_ident(block_arg, old_ptr, identity)
            op_result, idx = iterate_op.get_result_for_input_arg(use)
            self.propagate_ident(op_result, old_ptr, identity)
            yield_arg = iterate_op.get_yield_arg_for_result(op_result)
            yield_arg_type = cast(dlt.PtrType, yield_arg.type)
            if yield_arg_type.has_identity:
                self.ident_replacements[yield_arg_type.identification].add(identity)
                self.ident_replacements[identity].add(yield_arg_type.identification)









