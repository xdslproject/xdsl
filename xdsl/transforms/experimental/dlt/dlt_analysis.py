import typing
from typing import cast

from xdsl.dialects import func
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import IterIdent, PtrIdent
from xdsl.ir import Operation
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap, NestOp
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph


def analyse_read_only_ptrs(
    layout_graph: LayoutGraph, iter_map: IterationMap, new_orders: dict[IterIdent, dlt.IterationOrder], root_op: Operation
) -> tuple[set[PtrIdent], set[PtrIdent]]:

    # First collect every use
    uses: dict[
        NestOp,
        list[dlt.DTLLayoutScopedOp | func.Call],
    ] = {}
    order_parents: dict[
        NestOp|None, set[NestOp]
    ] = {}
    for child_op in root_op.walk():
        if isinstance(child_op, NestOp):
            parent_op = iter_map.parent_nest_for_op(child_op)
            order_parents.setdefault(parent_op, set()).add(child_op)
        if isinstance(child_op, dlt.DTLLayoutScopedOp | func.Call):
            child_op = cast(dlt.DTLLayoutScopedOp | func.Call, child_op)
            parent_op = iter_map.parent_nest_for_op(child_op)
            assert parent_op is not None
            uses.setdefault(parent_op, []).append(child_op)

    mutable: dict[NestOp, set[PtrIdent]] = {}
    for parent in set(uses.keys()) | set(order_parents.keys()):
        mutable[parent] = set()

    while True:
        changed = False

        for parent_nest, child_nests in order_parents.items():
            if not isinstance(parent_nest, dlt.IterateOp):
                continue
            for child_nest in child_nests:
                for mutable_sub_child in mutable.get(child_nest, set()):
                    if mutable_sub_child not in mutable.get(parent_nest, set()):
                        mutable.setdefault(parent_nest, set()).add(mutable_sub_child)
                        changed = True

        for parent_op, child_ops in uses.items():
            if isinstance(parent_op, func.FuncOp | dlt.LayoutScopeOp):
                continue
            for child_op in child_ops:
                if isinstance(child_op, dlt.SetOp):
                    child_op = typing.cast(dlt.SetOp, child_op)
                    ptr_type = cast(dlt.PtrType, child_op.tree.type)
                    if ptr_type.identification not in mutable.get(parent_op, set()):
                        mutable.setdefault(parent_op, set()).add(
                            ptr_type.identification
                        )
                        changed = True
                elif isinstance(child_op, dlt.SelectOp):
                    child_op = typing.cast(dlt.SelectOp, child_op)
                    input_ptr_type = cast(dlt.PtrType, child_op.tree.type)
                    output_ptr_type = cast(dlt.PtrType, child_op.res.type)
                    if input_ptr_type.identification in mutable.get(
                        parent_op, set()
                    ) and output_ptr_type.identification not in mutable.get(
                        parent_op, set()
                    ):
                        mutable.setdefault(parent_op, set()).add(
                            output_ptr_type.identification
                        )
                        changed = True
                elif isinstance(child_op, dlt.AllocOp):
                    child_op = typing.cast(dlt.AllocOp, child_op)
                    ptr_type = cast(dlt.PtrType, child_op.res.type)
                    if ptr_type.identification not in mutable.get(parent_op, set()):
                        mutable.setdefault(parent_op, set()).add(
                            ptr_type.identification
                        )
                        changed = True
                elif isinstance(child_op, dlt.DeallocOp):
                    child_op = typing.cast(dlt.DeallocOp, child_op)
                    ptr_type = cast(dlt.PtrType, child_op.tree.type)
                    if ptr_type.identification not in mutable.get(parent_op, set()):
                        mutable.setdefault(parent_op, set()).add(
                            ptr_type.identification
                        )
                        changed = True
                elif isinstance(child_op, func.Call):
                    for arg in [
                        arg
                        for arg in child_op.arguments
                        if isinstance(arg.type, dlt.PtrType)
                    ]:
                        ptr_type = cast(dlt.PtrType, arg.type)
                        if ptr_type.identification not in mutable.get(parent_op, set()):
                            mutable.setdefault(parent_op, set()).add(
                                ptr_type.identification
                            )
                            changed = True
        if not changed:
            break

    all_mutable = {}
    for nest, mutable_idents in mutable.items():
        if not isinstance(nest, dlt.IterateOp):
            continue
        bases: set[PtrIdent] = set()
        for ident in mutable_idents:
            current_bases = layout_graph.get_base_idents(ident)
            for base, ms, ds in current_bases:
                bases.add(base)
        new_mutable_idents: set[PtrIdent] = set()
        for base in bases:
            new_mutable_idents.update(layout_graph.get_transitive_closure(base))
        all_mutable[nest] = new_mutable_idents

    read_only_map: dict[IterIdent, set[PtrIdent]] = {}
    for iter_op, mutable_idents in all_mutable.items():
        ptr_idents = {arg.type.identification for arg in iter_op.get_block_args_for_tensor_args()}
        ptr_idents -= mutable_idents
        if iter_op.has_identity:
            read_only_map[iter_op.identification] = ptr_idents
    all_read_only = {ident for idents in read_only_map.values() for ident in idents}
    non_zero_reducibles = {}
    for iter_ident in read_only_map:
        order = new_orders[iter_ident]
        for ptr_ident in read_only_map[iter_ident]:
            iter_tensor_arg_idents = [arg.type.identification for arg in iter_map.iteration_ops[iter_ident].get_block_args_for_tensor_args()]
            if ptr_ident in iter_tensor_arg_idents:
                tensor_idx = iter_tensor_arg_idents.index(ptr_ident)
                is_non_zero_reducible = order.non_zero_loop_for(tensor_idx, iter_map.iteration_ops[iter_ident])
                if is_non_zero_reducible:
                    non_zero_reducibles.setdefault(iter_ident, set()).add(ptr_ident)
    all_non_zero_reducibles = {ident for idents in non_zero_reducibles.values() for ident in idents}

    return all_read_only, all_non_zero_reducibles
