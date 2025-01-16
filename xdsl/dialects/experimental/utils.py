from xdsl.builder import Builder
from xdsl.dialects import affine, arith, builtin, func, linalg, memref
from xdsl.dialects.experimental.hida_functional import DispatchOp, TaskOp, YieldOp
from xdsl.dialects.experimental.hida_structural import NodeOp, ScheduleOp
from xdsl.ir import BlockArgument, Use
from xdsl.irdl import Block, Operation
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.transforms.experimental.liveness import Liveness

# from xdsl.traits import MemoryEffect, MemoryEffectKind, get_effects
from xdsl.utils.hints import isa


def dispatch_block(block: Block):
    if list(filter(lambda x: isinstance(x, DispatchOp), block.ops)) or not isinstance(
        block.parent_op(), func.FuncOp | affine.For
    ):
        return DispatchOp([], [])

    assert block.last_op
    return_values = list(block.last_op.operands)

    yield_op = YieldOp(*return_values)
    dispatch_ops: list[Operation] = []
    for op in block.ops:
        if op != block.last_op:
            op.detach()
            dispatch_ops.append(op)

    dispatch_ops.append(yield_op)
    return_types = [rval.type for rval in return_values]

    dispatch = DispatchOp(dispatch_ops, return_types)
    block.insert_op_before(dispatch, block.last_op)

    return dispatch


def is_element_wise_generic_op(op: linalg.Generic):
    assert isinstance(op, linalg.Generic)
    if op.get_num_loops() != sum(
        list(map(lambda x: (1 if x.data == "parallel" else 0), op.iterator_types.data))
    ):
        return

    for value_map in zip(op.operands, op.get_indexing_maps()):
        operand_type = value_map[0].type
        affine_map: builtin.AffineMap = value_map[1]

        # TODO: check if the type is static

        assert isinstance(operand_type, builtin.TensorType)

        index = affine_map.num_dims - operand_type.get_num_dims()
        for shape_expr in zip(operand_type.shape, affine_map.results):
            dim_size = shape_expr[0].data
            expr = shape_expr[1]

            if expr != builtin.AffineDimExpr(index) and dim_size != 1:
                return False
            index += 1

    return True


def fuse_ops_into_task(
    ops: list[Operation], rewriter: PatternRewriter, insert_to_last_op: bool = False
):
    # The output values of the task are the output values of the member operations that have uses outside the task
    all_nested_ops = set([sub_op for op in ops for sub_op in op.walk()])
    output_values = list(
        filter(
            lambda x: any([use.operation not in all_nested_ops for use in x.uses]),
            [res for op in ops for res in op.results],
        )
    )
    task_res_types = list(map(lambda x: x.type, output_values))

    if insert_to_last_op:
        insert_point = InsertPoint.after(ops[-1])
    else:
        insert_point = InsertPoint.before(ops[0])

    yield_op = YieldOp(*output_values)

    task = TaskOp([], task_res_types)
    rewriter.insert_op(task, insert_point)

    list(map(lambda x: x.detach(), ops))
    # list(map(lambda x: task.region.block.add_op(x), ops))
    rewriter.insert_op(ops, InsertPoint.at_end(task.region.block))
    # task.region.block.add_op(yield_op)
    rewriter.insert_op(yield_op, InsertPoint.at_end(task.region.block))

    # Propagate output values of the task to users of the outputs of the member operations
    for res_idx, res in enumerate(task.results):
        output_values[res_idx].replace_by_if(
            res, lambda use: not task.is_ancestor(use.operation)
        )

    # Inline all sub tasks
    for subtask in filter(lambda x: isinstance(x, TaskOp), ops):
        assert isinstance(subtask, TaskOp)
        assert isinstance(subtask.region.block.last_op, Operation)

        yield_operands = list(
            map(lambda x: x.owner, subtask.region.block.last_op.operands)
        )
        assert isa(yield_operands, list[Operation])

        subtask_ops: list[Operation] = [sub_op for sub_op in subtask.region.ops]
        list(map(lambda x: x.detach(), subtask_ops))
        rewriter.replace_op(
            subtask,
            subtask_ops[:-1],
            new_results=[
                res for yield_operand in yield_operands for res in yield_operand.results
            ],
        )


def get_child_loop_num(op: Operation):
    n_children = 0

    for region in op.regions:
        for block in region.blocks:
            for sub_op in block.ops:
                if isinstance(sub_op, affine.For):
                    n_children += 1

    return n_children


def get_loop_band_from_innermost(for_op: affine.For):
    band: list[affine.For] = []

    reverse_band: list[affine.For] = []

    current_loop = for_op

    while True:
        reverse_band.append(current_loop)

        parent_loop = current_loop.get_parent_of_type(affine.For)
        if not parent_loop:
            break

        if get_child_loop_num(parent_loop) == 1:
            current_loop = parent_loop
        else:
            break

    band += reversed(reverse_band)
    return band


# NOTE: this is the method used in HIDA for partitioning of affine: it only considers
# bands where loop nests have one or more children
def get_loop_bands(
    block: Block, bands: list[list[affine.For]], allow_having_children: bool
):
    bands.clear()

    for loop in filter(lambda x: isinstance(x, affine.For), block.walk()):
        assert isinstance(loop, affine.For)
        n_children = get_child_loop_num(loop)

        if n_children == 0 or (n_children > 1 and allow_having_children):
            bands.append(get_loop_band_from_innermost(loop))


def get_loop_bands_any_nchildren(block: Block):
    bands: list[list[affine.For]] = []

    for loop in filter(lambda x: isinstance(x, affine.For), block.walk()):
        bands.append(get_loop_band_from_innermost(loop))

    return bands


def is_fully_contained_in_band(op: Operation, band: list[affine.For]):
    for operand in op.operands:
        outer_band_loop = band[0]
        if not outer_band_loop.is_ancestor(operand.owner):
            return False

    return True


def get_minimum_independent_band(bands: list[list[affine.For]]):
    for band in reversed(bands):
        is_minimum_band = True

        # NOTE: this is assuming perfect loops for simplicity for now.
        inner_band_loop = bands[-1]
        for op in inner_band_loop:
            if not is_fully_contained_in_band(op, band):
                is_minimum_band = False

        if is_minimum_band:
            return band

    return None


# A store is invariant in a loop band when it is not modified in any of the iterations. This
# property holds when the induction variable of the top loop of the band is not the lowermost index
# of the store.
def is_store_invariant_in_band(store: affine.Store, band: list[affine.For]):
    innermost_band_loop = band[-1]
    lowermost_index = store.indices[-1]

    if lowermost_index == innermost_band_loop.body.block.args[0]:
        return False

    return True


def get_invariant_output_band(bands: list[list[affine.For]]):
    # This method assumes the bands contain a convolution or similar local reduction pattern.
    # Returns the band where the output store is invariant in the innermost band loop.
    outermost_loop = bands[0][0]
    store = [op for op in outermost_loop.walk() if isinstance(op, affine.Store)][0]

    for band in bands:
        if is_store_invariant_in_band(store, band):
            return band

    return None


def hoist_constants(node: func.FuncOp, rewriter: PatternRewriter):
    all_constants = [op for op in node.walk() if isinstance(op, arith.Constant)]

    for constant in all_constants:
        constant.detach()
        rewriter.insert_op(constant, InsertPoint.before(node))


def tile_loop(
    loop: affine.For,
    index: int,
    factor: int,
    rewriter: PatternRewriter,
    parent_node: func.FuncOp,
):
    # TODO: For now the tiling is happening at the outermost loop
    # TODO: assuming for now that the bounds are constant. Also assuming step=1
    lb = loop.lowerBoundMap.get_affine_map().results[0].value
    ub = loop.upperBoundMap.get_affine_map().results[0].value
    n_subloops = factor  # TODO: assuming even division for now

    sub_len = int((ub - lb) / factor)
    sub_lb = lb
    sub_ub = lb + sub_len

    sub_loop_idx = 0
    sub_loops_lst: list[affine.For] = []
    sub_loop_nodes_lst: list[func.FuncOp] = []

    # Convert into a list to guarantee ordering
    liveins = list(Liveness(loop).get_livein(loop.body.block))
    external_liveins = [
        livein
        for livein in liveins
        if any(
            [
                loop.is_ancestor(use.operation) and not loop.is_ancestor(livein.owner)
                for use in livein.uses
            ]
        )
    ]
    in_types = [livein.type for livein in external_liveins]

    for _ in range(n_subloops):
        loop_clone = loop.clone()
        sub_region = rewriter.move_region_contents_to_new_regions(loop_clone.body)

        sub_loop = affine.For.from_region((), (), (), (), sub_lb, sub_ub, sub_region)
        sub_loops_lst.append(sub_loop)
        sub_lb += sub_len
        sub_ub += sub_len

        rewriter.insert_op(sub_loop, InsertPoint.before(loop))

        @Builder.region(in_types)
        def node_body(builder: Builder, args: list[BlockArgument, ...]):
            sub_loop.detach()
            builder.insert(sub_loop)
            for livein, func_arg in zip(external_liveins, args):
                livein.replace_by_if(
                    func_arg, lambda use: sub_loop.is_ancestor(use.operation)
                )

            builder.insert(func.Return())

        sub_loop_node = func.FuncOp(
            f"sub_node_{parent_node.sym_name.data}_{sub_loop_idx}",
            builtin.FunctionType.from_lists(in_types, []),
            node_body,
        )
        sub_loop_nodes_lst.append(sub_loop_node)
        sub_loop_idx += 1
        sub_loop_node.attributes["top_func"] = builtin.UnitAttr()
        sub_loop_node.attributes["original_node"] = (
            parent_node.sym_name
        )  # builtin.IntegerAttr.from_int_and_width(0, 32)

        rewriter.insert_op(sub_loop_node, InsertPoint.before(parent_node))

    @Builder.region(in_types)
    def parent_node_body(builder: Builder, args: list[BlockArgument, ...]):
        for sub_loop_node in sub_loop_nodes_lst:
            builder.insert(func.Call(sub_loop_node.sym_name.data, args, []))
        builder.insert(func.Return())

    top = func.FuncOp(
        parent_node.sym_name.data,
        builtin.FunctionType.from_lists(in_types, []),
        parent_node_body,
    )
    rewriter.replace_matched_op(top)
    top.attributes["TOP"] = builtin.UnitAttr()


def is_written(use: Use):
    # For ScheduleOp, we don't rely on memory effect interface. Instead, we delve
    # into its region to figure out the effect. However, for NodeOp, we don't
    # need this recursive approach any more.
    if isinstance(use.operation, NodeOp):
        # TODO
        #    return node.get_operand_kind(use) == OperandKind.OUTPUT
        return
    elif isinstance(schedule := use.operation, ScheduleOp):
        return any(
            map(lambda x: is_written(x), schedule.region.block.args[use.index].uses)
        )
    # TODO: elif viewlikeopinterface

    # mem_effects = list(filter(lambda x: isinstance(x, MemoryEffect), get_effects(use.operation)))
    # effects = get_effects(use.operation)

    # if effects:  # TODO: check for streams too
    #    mem_effects = list(filter(lambda x: isinstance(x, MemoryEffect), effects))
    #    print("MEM EFFECTS: ", mem_effects)
    #    for mem_effect in mem_effects:
    #        if mem_effect.kind == MemoryEffectKind.WRITE:
    #            return True
    # FIXME: for now we will look at the operation type instead of the side effects, since these are not implemented in xDSL yet
    if isinstance(use.operation, affine.Store | memref.Store):
        return True
    return False
