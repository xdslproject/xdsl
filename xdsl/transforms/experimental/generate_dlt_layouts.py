"""
Lower Data-Layout Trees into LLVM struct types (and other things)
"""

import abc
import typing
from dataclasses import dataclass
from typing import Iterable, Self, TypeVar, cast

from xdsl.dialects import arith, builtin, func, llvm, printf
from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr
from xdsl.dialects.experimental import dlt
from xdsl.ir import (
    Attribute,
    BlockArgument,
    MLContext,
    Operation,
    OpResult,
    Region,
    SSAValue,
)
from xdsl.irdl import Operand
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.printer import Printer


@dataclass(frozen=True)
class ElementsUse:
    op: Operation
    operand: Operand
    member_specifers: frozenset[dlt.MemberAttr]
    dimensions: frozenset[dlt.DimensionAttr]


class Trace(abc.ABC):
    parent: "TraceNode"

    def __init__(self, parent: "TraceNode") -> None:
        self.parent = parent

    @staticmethod
    def base_node() -> "TraceNode":
        return TraceNode([], [], None)

    @abc.abstractmethod
    def constraints(self) -> tuple[set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        pass


class TraceNode(Trace):
    member_specifiers: set[dlt.MemberAttr]
    dimensions: set[dlt.DimensionAttr]
    children: list[Trace]

    def __init__(
        self,
        member_specifiers: Iterable[dlt.MemberAttr],
        dimensions: Iterable[dlt.DimensionAttr],
        parent: "TraceNode",
    ) -> None:
        self.member_specifiers = set(member_specifiers)
        self.dimensions = set(dimensions)
        self.children = []
        super().__init__(parent)

    def child_like(
        self,
        member_specifiers: Iterable[dlt.MemberAttr],
        dimensions: Iterable[dlt.DimensionAttr],
    ) -> Self:
        member_specifiers = set(member_specifiers)
        dimensions = set(dimensions)
        for child in self.children:
            if isinstance(child, TraceNode):
                if (
                    child.member_specifiers == member_specifiers
                    and child.dimensions == dimensions
                ):
                    return child
        new_node = TraceNode(member_specifiers, dimensions, self)
        self.children.append(new_node)
        return new_node

    def add_leaf(self, elem_use: ElementsUse):
        new_node = TraceLeaf(elem_use, self)
        self.children.append(new_node)
        return new_node

    def constraints(self) -> tuple[set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        if self.parent is not None:
            p_ms, p_ds = self.parent.constraints()
        else:
            p_ms, p_ds = set(), set()
        return p_ms | self.member_specifiers, p_ds | self.dimensions


class TraceLeaf(Trace):
    use: ElementsUse

    def __init__(self, use: ElementsUse, parent: "TraceNode") -> None:
        self.use = use
        super().__init__(parent)

    def constraints(self) -> tuple[set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        return self.parent.constraints()


@dataclass
class DLTLayoutRewriter(RewritePattern):

    def _get_base(self, operand: SSAValue) -> SSAValue:
        assert isinstance(operand.type, dlt.PtrType)
        if isinstance(operand, OpResult):
            if isinstance(operand.op, dlt.SelectOp):
                return self._get_base(operand.op.tree)
            elif isinstance(operand.op, dlt.AllocOp):
                return operand.op.res
            else:
                assert False
        elif isinstance(operand, BlockArgument):
            return operand
        else:
            assert False

    def _for_each_op(self, regions: list[Region], func):
        regions: list[Region] = regions.copy()
        while regions:
            region = regions.pop()
            for block in region.blocks:
                for op in block.ops:
                    regions += op.regions
                    func(op)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, scope: dlt.LayoutScopeOp, rewriter: PatternRewriter):
        scope.verify_()
        print("SCOPE REWRITE")

        module = scope.parent.parent.parent
        while not isinstance(module, ModuleOp):
            module = module.parent.parent.parent

        funcs: dict[StringAttr, tuple[func.FuncOp, list[func.Call]]] = (
            scope.get_function_map()
        )

        layout_entry_points: list[tuple[Operation, int]] = (
            []
        )  # int == -1 for AllocOp, and the argument index of the PtrType for FuncOp
        layout_entry_point_operands: list[SSAValue] = []
        entry_point_layouts: list[dlt.Layout] = (
            []
        )  # int == -1 for AllocOp, and the argument index of the PtrType for FuncOp
        named_abstract_layouts: dict[str, tuple[dlt.Layout, set[int]]] = (
            {}
        )  # ints are indices into layout_entry_points
        for op in scope.walk():
            if isinstance(op, dlt.AllocOp):
                op: dlt.AllocOp = op
                assert isinstance(op.res.type, dlt.PtrType)
                for layout in op.res.type.layout.walk():
                    if isinstance(layout, dlt.NamedLayoutAttr):
                        l, s = named_abstract_layouts.setdefault(
                            layout.abstract_name.data, (layout, set())
                        )
                        if l != layout:
                            assert False
                        assert l == layout
                        s.add(len(layout_entry_points))
                layout_entry_points.append((op, -1))
                layout_entry_point_operands.append(op.res)
                entry_point_layouts.append(op.res.type.layout)
            elif isinstance(op, func.FuncOp):
                op: func.FuncOp = op
                for i, input in enumerate(op.function_type.inputs):
                    if isinstance(input, dlt.PtrType):
                        for layout in input.layout.walk():
                            if isinstance(layout, dlt.NamedLayoutAttr):
                                l, s = named_abstract_layouts.setdefault(
                                    layout.abstract_name.data, (layout, set())
                                )
                                assert l == layout
                                s.add(len(layout_entry_points))
                        layout_entry_points.append((op, i))
                        layout_entry_point_operands.append(op.args[i])
                        entry_point_layouts.append(input.layout)

        if not any(layout.is_abstract for layout in entry_point_layouts):
            print("No abstract layouts to specify")
            # return

        uses_map: dict[int, set[ElementsUse]] = {}
        element_uses_map: dict[tuple[int, dlt.ElementAttr], set[ElementsUse]] = {}
        for op in scope.walk():
            if isinstance(op, dlt.GetOp | dlt.SetOp):
                bases = _get_deep_base(op.tree, funcs)
                for base, ms, ds in bases:
                    use = ElementsUse(op, op.tree, ms, ds)
                    base_idx = layout_entry_point_operands.index(base)
                    uses_map.setdefault(base_idx, set()).add(use)
                    base_ptr_type = base.type
                    assert isinstance(base_ptr_type, dlt.PtrType)
                    ptr_type = cast(dlt.PtrType, base_ptr_type)
                    if isinstance(op, dlt.GetOp):
                        base_type = op.get_type
                    else:
                        assert isinstance(op, dlt.SetOp)
                        base_type = op.set_type
                    elem_type_type = ptr_type.contents_type.with_selection(
                        ms, ds, base_type
                    )
                    elem = elem_type_type.get_single_element()
                    assert elem is not None
                    element_uses_map.setdefault((base_idx, elem), set()).add(use)
            elif isinstance(op, dlt.CopyOp):
                src_bases = _get_deep_base(op.src, funcs)
                for src_base, src_ms, src_ds in src_bases:
                    use = ElementsUse(op, op.src, src_ms, src_ds)
                    src_base_idx = layout_entry_point_operands.index(src_base)
                    uses_map.setdefault(src_base_idx, set()).add(use)
                    elem_type_type = src_base.type.contents_type.with_selection(
                        src_ms, src_ds, op.copy_type
                    )
                    for elem in elem_type_type.elements:
                        element_uses_map.setdefault((src_base_idx, elem)).add(use)
                dst_bases = _get_deep_base(op.dst, funcs)
                for dst_base, dst_ms, dst_ds in dst_bases:
                    uses_map.setdefault(
                        layout_entry_point_operands.index(dst_base), set()
                    ).add(ElementsUse(op, op.dst, dst_ms, dst_ds))
            elif isinstance(op, dlt.AllocOp):
                for initialValue in op.initialValues:
                    bases = _get_deep_base(initialValue, funcs)
                    for base, ms, ds in bases:
                        uses_map.setdefault(
                            layout_entry_point_operands.index(base), set()
                        ).add(ElementsUse(op, initialValue, ms, ds))

        use_traces: dict[int, TraceNode] = {}

        def add_traces(
            operand: Operand, trace: TraceNode, seen_funcs: set[func.FuncOp] = set()
        ):
            for use in operand.uses:
                if isinstance(use.operation, dlt.SelectOp):
                    child_node = trace.child_like(
                        use.operation.members, use.operation.dimensions
                    )
                    add_traces(use.operation.res, child_node, seen_funcs)
                elif isinstance(use.operation, dlt.IterateOp):
                    op: dlt.IterateOp = use.operation
                    block_arg, dims = op.get_block_arg_and_dims_for_input_arg(use)
                    if dims is None:
                        # This is an Iter_arg
                        add_traces(block_arg, trace, seen_funcs)
                    else:
                        child_node = trace.child_like(
                            [], {dim for ds in dims for dim in ds}
                        )
                        add_traces(block_arg, child_node, seen_funcs)
                elif isinstance(use.operation, dlt.IterateYieldOp):
                    iter_op = use.operation.parent_op()
                    assert isinstance(iter_op, dlt.IterateOp)
                    res = iter_op.get_result_for_yield_use()
                    add_traces(res, trace, seen_funcs)
                elif isinstance(
                    use.operation, dlt.GetOp | dlt.SetOp | dlt.ExtractExtentOp
                ):
                    ms, ds = trace.constraints()
                    bases = _get_deep_base(operand, funcs)
                    for base, b_ms, b_ds in bases:
                        assert frozenset(ms) == b_ms
                        assert frozenset(ds) == b_ds
                    elem_use = ElementsUse(
                        use.operation, operand, frozenset(ms), frozenset(ds)
                    )
                    trace.add_leaf(elem_use)
                elif isinstance(use.operation, dlt.CopyOp):
                    ms, ds = trace.constraints()
                    elem_use = ElementsUse(
                        use.operation, operand, frozenset(ms), frozenset(ds)
                    )
                    trace.add_leaf(elem_use)
                elif isinstance(use.operation, func.Call):
                    op, calls = funcs[use.operation.callee.root_reference]
                    if op not in seen_funcs:
                        new_seen_funcs = seen_funcs | {op}
                        add_traces(op.body.block.args[use.index], trace, new_seen_funcs)
                elif isinstance(use.operation, func.Return):
                    func_op = use.operation.parent_op()
                    assert isinstance(func_op, func.FuncOp)
                    if (
                        func_op.sym_visibility is None
                        or func_op.sym_visibility.data != "private"
                    ):
                        ms, ds = trace.constraints()
                        elem_use = ElementsUse(
                            use.operation, operand, frozenset(ms), frozenset(ds)
                        )
                        trace.add_leaf(elem_use)
                    op, calls = funcs[func_op.sym_name]
                    for call in calls:
                        add_traces(call.results[use.index], trace, seen_funcs)
                else:
                    raise NotImplementedError(
                        f"Not implemented for type: {type(use.operation)} : {use.operation}"
                    )

        for i, operand in enumerate(layout_entry_point_operands):
            use_traces[i] = Trace.base_node()
            add_traces(operand, use_traces[i])

        # print(layout_entry_points)
        # print(uses_map)

        name_map = {}
        new_entry_point_layouts = _make_dense_layouts(entry_point_layouts, name_map)

        def propergate_operands(operand: SSAValue):
            assert isinstance(operand.type, dlt.PtrType)
            for use in operand.uses:
                if isinstance(use.operation, dlt.SelectOp):
                    op: dlt.SelectOp = use.operation
                    assert operand == op.operands[use.index]
                    current_ident = cast(dlt.PtrType, op.res.type).identification
                    new_res_type = dlt.SelectOp.calculateResultType(
                        operand.type, op.members, op.dimensions
                    ).as_not_base().with_identification(current_ident)
                    if new_res_type != op.res.type:
                        new_op = dlt.SelectOp(
                            op.tree,
                            op.members,
                            op.dimensions,
                            op.values,
                            result_type=new_res_type,
                        )
                        rewriter.replace_op(op, new_op)
                        propergate_operands(new_op.res)
                elif isinstance(use.operation, dlt.IterateOp):
                    op: dlt.IterateOp = use.operation
                    assert operand == op.operands[use.index]
                    block = op.body.block
                    block_arg, dims = op.get_block_arg_and_dims_for_input_arg(use)
                    assert isinstance(block_arg.type, dlt.PtrType)
                    if dims is None:
                        # This is an Iter_arg
                        result, idx = op.get_result_for_input_arg(use)
                        if operand.type != block_arg.type:
                            rewriter.modify_block_argument_type(block_arg, operand.type)
                            propergate_operands(block_arg)

                        yield_op = block.last_op
                        assert isinstance(yield_op, dlt.IterateYieldOp)
                        yielded = yield_op.arguments[idx]
                        assert yielded.type == operand.type
                        if result.type != operand.type:
                            result.type = operand.type
                            rewriter.handle_operation_modification(op)
                            # We then need to propagate beyond the IterateOp
                            propergate_operands(result)
                    else:
                        # this is a tensor arg, so there are dimensions to select.
                        selected_dims = [d for ds in dims for d in ds]
                        current_ident = block_arg.type.identification
                        new_inner_type = dlt.SelectOp.calculateResultType(
                            operand.type, [], selected_dims
                        ).as_not_base().with_identification(current_ident)
                        assert (
                            block_arg.type.contents_type == new_inner_type.contents_type
                        )
                        if block_arg.type != new_inner_type:
                            rewriter.modify_block_argument_type(
                                block_arg, new_inner_type
                            )
                            propergate_operands(block_arg)
                elif isinstance(use.operation, dlt.GetOp):
                    pass
                elif isinstance(use.operation, dlt.SetOp):
                    pass
                elif isinstance(use.operation, dlt.CopyOp):
                    pass
                elif isinstance(use.operation, dlt.ExtractExtentOp):
                    pass
                elif isinstance(use.operation, func.FuncOp):
                    raise NotImplementedError(
                        f"cannot propagate dlt ptr type through {use.operation}"
                    )
                elif isinstance(use.operation, func.Call):
                    call_op = cast(func.Call, use.operation)
                    func_name = call_op.callee.root_reference
                    if func_name not in funcs:
                        raise NotImplementedError(
                            f"cannot propagate dlt ptr type through call to unknown function: {use.operation}"
                        )
                    func_op, calls = funcs[func_name]
                    if func_op.args[use.index].type != operand.type:
                        func_op.replace_argument_type(use.index, operand.type)
                        propergate_operands(func_op.args[use.index])
                    # raise NotImplementedError(f"cannot propagate dlt ptr type through {use.operation}")
                elif isinstance(use.operation, func.Return):
                    func_op = use.operation.parent_op()
                    assert isinstance(func_op, func.FuncOp)
                    func_op.update_function_type()
                    op, calls = funcs[func_op.sym_name]
                    for call in calls:
                        call_operand = call.results[use.index]
                        call_operand.type = operand.type
                        rewriter.handle_operation_modification(call)
                        propergate_operands(call_operand)
                else:
                    raise NotImplementedError(
                        f"cannot propagate dlt ptr type through {use.operation}"
                    )

        def update_and_propagate(operation: tuple[Operation, int], layout: dlt.Layout):
            op, idx = operation
            if isinstance(op, dlt.AllocOp):
                op: dlt.AllocOp = op
                op_res_type: dlt.PtrType = op.res.type
                assert idx == -1
                new_contents = layout.contents_type
                if op_res_type.contents_type != new_contents:
                    raise ValueError(
                        f"New layout is incompatible with existing type. Expected type {op_res_type.contents_type} but got {new_contents} from {layout}"
                    )
                new_ptr_type = op_res_type.with_new_layout(layout, preserve_ident=True)
                if op_res_type != new_ptr_type:
                    new_alloc_op = dlt.AllocOp(new_ptr_type,
                                               op.init_extent_mapping(),
                                               op.initialValues)
                    rewriter.replace_op(op, new_alloc_op)
                    propergate_operands(new_alloc_op.res)
            elif isinstance(op, func.FuncOp):
                op: func.FuncOp = op
                function_type = op.function_type
                new_contents = layout.contents_type
                block_arg = op.args[idx]
                assert isinstance(block_arg.type, dlt.PtrType)
                if block_arg.type.contents_type != new_contents:
                    raise ValueError(
                        f"New layout is incompatible with existing type. Expected type {function_type.inputs.data[idx].contents_type} but got {new_contents} from {layout}"
                    )
                new_type = block_arg.type.with_new_layout(layout, preserve_ident=True)
                if block_arg.type != new_type:
                    op.replace_argument_type(block_arg, new_type)
                    propergate_operands(block_arg)
            else:
                raise NotImplementedError()

        for entry, layout in zip(layout_entry_points, new_entry_point_layouts):
            update_and_propagate(entry, layout)
        print("DONE")


T = TypeVar("T", dlt.Layout, list[dlt.Layout])


def _make_dense_layouts(layout: T, map: dict[str, dlt.Layout]) -> T:
    if isinstance(layout, list):
        return [_make_dense_layouts(e, map) for e in layout]
    # elif isinstance(layout, dlt.NamedLayoutAttr):
    #     layout: dlt.NamedLayoutAttr = layout
    #     if layout.abstract_name.data in map:
    #         return map[layout.abstract_name.data]
    #     else:
    #         sub_layout = _make_dense_layouts(layout.child, map)
    #         new_layout = dlt.NamedLayoutAttr(layout.abstract_name, sub_layout)
    #         assert new_layout.abstract_name.data not in map
    #         map[new_layout.abstract_name.data] = new_layout
    #         return new_layout
    elif isinstance(layout, dlt.AbstractLayoutAttr):
        layout: dlt.AbstractLayoutAttr = layout
        sub_layouts = []
        for child in layout.children:
            sub_layout = _make_dense_layouts(child.child, map)
            for dim in list(child.dimensions):
                sub_layout = dlt.DenseLayoutAttr(sub_layout, dim)
            for member in list(child.member_specifiers):
                sub_layout = dlt.MemberLayoutAttr(sub_layout, member)
            sub_layouts.append(sub_layout)
        if len(sub_layouts) == 1:
            sub_layout = sub_layouts[0]
        else:
            assert len(sub_layouts) > 1
            sub_layout = dlt.StructLayoutAttr(sub_layouts)
        return sub_layout
    else:
        children = [_make_dense_layouts(child, map) for child in layout.get_children()]
        return layout.from_new_children(children)


def _try_apply_sparse(layout: dlt.Layout):
    if isinstance(layout, list):
        return [_try_apply_sparse(l) for l in layout]
    if isinstance(layout, dlt.AbstractLayoutAttr):
        a_layout = typing.cast(dlt.AbstractLayoutAttr, layout)
        if len(dims := a_layout.common_abstract_dimensions()) > 1:
            dims = list(dims)
            dims.sort(key= lambda d: d.extent.value.value.data if isinstance(d.extent, dlt.StaticExtentAttr) else 0)
            sparse = [dims.pop()]
            return _make_sparse_layout(a_layout, dims, sparse)
    children = [_try_apply_sparse(child) for child in layout.get_children()]
    return layout.from_new_children(children)


def _make_sparse_layout(layout: dlt.AbstractLayoutAttr, direct_dims: list[dlt.DimensionAttr], sparse_dims: list[dlt.DimensionAttr]):
    assert layout.common_abstract_dimensions().issuperset(direct_dims+sparse_dims)
    assert all(all(dim in child.dimensions for dim in sparse_dims) for child in layout.children)
    assert len(set(direct_dims).intersection(set(sparse_dims))) == 0
    direct_node = dlt.AbstractLayoutAttr([([],direct_dims,dlt.PrimitiveLayoutAttr(dlt.IndexRangeType()))])
    abstract_children = [dlt.AbstractChildAttr(a_child.member_specifiers, a_child.dimensions.remove(direct_dims+sparse_dims), a_child.child) for a_child in layout.children]
    coo_node = dlt.UnpackedCOOLayoutAttr(dlt.AbstractLayoutAttr(abstract_children), sparse_dims)
    return dlt.IndexingLayoutAttr(direct_node, coo_node)


def _get_deep_base(
    operand: SSAValue,
    func_map: dict[StringAttr, tuple[func.FuncOp, list[func.Call]]],
    func_ops_seen: frozenset[func.FuncOp] = frozenset(),
) -> set[tuple[SSAValue, frozenset[dlt.MemberAttr], frozenset[dlt.DimensionAttr]]]:
    assert isinstance(operand.type, dlt.PtrType)
    if isinstance(operand, OpResult):
        op = operand.op
        if isinstance(op, dlt.AllocOp):
            return {(op.res, frozenset(), frozenset())}
        elif isinstance(op, dlt.SelectOp):
            inits = _get_deep_base(op.tree, func_map, func_ops_seen)
            return {
                (
                    init,
                    ms.union({m for m in op.members}),
                    ds.union({d for d in op.dimensions}),
                )
                for (init, ms, ds) in inits
            }
        elif isinstance(op, dlt.IterateOp):
            yield_arg = op.get_yield_arg_for_result(operand)
            return _get_deep_base(yield_arg, func_map, func_ops_seen)
        else:
            assert False
    elif isinstance(operand, BlockArgument):
        parent_op = operand.block.parent_op()
        arg_index = operand.index
        if isinstance(parent_op, dlt.IterateOp):
            input_arg = parent_op.get_input_arg_for_block_arg(operand)
            return _get_deep_base(input_arg, func_map, func_ops_seen)
        elif isinstance(parent_op, func.FuncOp):
            op, calls = func_map[parent_op.sym_name]
            args = [
                call.arguments[arg_index]
                for call in calls
                if parent_op not in func_ops_seen
            ]
            func_ops_seen = func_ops_seen | frozenset([parent_op])
            if parent_op.sym_visibility == StringAttr("public"):
                public_arg = {(operand, frozenset(), frozenset())}
            else:
                public_arg = set()
            results = {
                a for arg in args for a in _get_deep_base(arg, func_map, func_ops_seen)
            } | public_arg
            if (
                len(results) > 1
            ):  # If a func has more than one entry point we must ensure they are named
                layout_name = None
                layout = None
                for arg in args + ([operand] if public_arg else []):
                    assert isinstance(arg.type, dlt.PtrType)
                    assert arg.type.has_identity
                    name = arg.type.identification
                    assert layout_name is None or layout_name == name
                    layout_name = name
                    assert layout is None or layout == arg.type.layout
                    layout = arg.type.layout
            return results
        else:
            assert False
    else:
        assert False
