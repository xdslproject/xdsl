from xdsl.dialects.experimental.hida_functional import TaskOp, DispatchOp, YieldOp
from xdsl.irdl import Operation, Block, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl.rewriter import InsertPoint
from xdsl.dialects import affine, linalg, builtin
from xdsl.builder import Builder


def dispatch_block(block : Block):
    # TODO: support for return values
    #if list(filter(lambda x: isinstance(x, DispatchOp), block.ops)) or not any(map(lambda x: isinstance(x, func.FuncOp) or isinstance(x, affine.For), block.ops)):
    #    return DispatchOp([])
    
    #return_values = list(map(lambda x: x.owner, [block.get_terminator().operands]))

    #assert isinstance(block.get_terminator(), Operation)
    block_terminator = block.get_terminator()
    assert isinstance(block_terminator, Operation)
    return_values : list[Operation] = [res.op for operand in block_terminator.operands for res in operand.owner.results]
    dispatch = DispatchOp(return_values)

    dispatch_block = dispatch.region.block
    builder = Builder.at_end(dispatch_block)
    builder.insert(YieldOp(*return_values))

    for op in list(block.ops)[:-1]:
        op.detach()
        assert isinstance(dispatch_block.last_op, Operation)
        dispatch_block.insert_op_before(op, dispatch_block.last_op)

    assert isinstance(block.last_op, Operation)
    block.insert_op_before(dispatch, block.last_op)

    return dispatch

def is_element_wise_generic_op(op : linalg.Generic):
    assert isinstance(op, linalg.Generic)
    if op.get_num_loops() != sum(list(map(lambda x: (1 if x.data == "parallel" else 0), op.iterator_types.data))):
        return
    
    for value_map in zip(op.operands, op.get_indexing_maps()):
        operand_type = value_map[0].type
        affine_map : builtin.AffineMap = value_map[1]

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

def fuse_ops_into_task(ops : list[Operation], rewriter: PatternRewriter, insert_to_last_op : bool = False):
    # The output values of the task are the output values of the member operations that have uses outside the task
    all_nested_ops = set([sub_op for op in ops for sub_op in op.walk()])
    output_values = list(filter(lambda x: any([not use.operation in all_nested_ops for use in x.uses]), [res for op in ops for res in op.results]))
    task_res_types = list(map(lambda x: x.type, output_values))

    if insert_to_last_op:
        insert_point = InsertPoint.after(ops[-1])
    else:
        insert_point = InsertPoint.before(ops[0])


    yield_op = YieldOp(*output_values)

    task = TaskOp([], task_res_types)
    rewriter.insert_op(task, insert_point)

    list(map(lambda x: x.detach(), ops))
    #list(map(lambda x: task.region.block.add_op(x), ops))
    rewriter.insert_op(ops, InsertPoint.at_end(task.region.block))
    #task.region.block.add_op(yield_op)
    rewriter.insert_op(yield_op, InsertPoint.at_end(task.region.block))

    # Propagate output values of the task to users of the outputs of the member operations
    for res_idx,res in enumerate(task.results):
        output_values[res_idx].replace_by_if(res, lambda use: not task.is_ancestor(use.operation))

    # Inline all sub tasks
    for subtask in filter(lambda x: isinstance(x, TaskOp), ops):
        assert isinstance(subtask, TaskOp) and isinstance(subtask.region.block.last_op, Operation)

        yield_operands = list(map(lambda x: x.owner, subtask.region.block.last_op.operands))
        assert isa(yield_operands, list[Operation])

        subtask_ops : list[Operation] = [sub_op for sub_op in subtask.region.ops]
        list(map(lambda x: x.detach(), subtask_ops))
        rewriter.replace_op(subtask, subtask_ops[:-1], new_results=[res for yield_operand in yield_operands for res in yield_operand.results])



def get_child_loop_num(op : Operation):
    n_children = 0

    for region in op.regions:
        for block in region.blocks:
            for sub_op in block.ops:
                if isinstance(sub_op, affine.For):
                    n_children += 1

    return n_children

def get_loop_band_from_innermost(for_op : affine.For):
    band : list[affine.For] = []

    reverse_band : list[affine.For] = []

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

def get_loop_bands(block : Block, bands : list[list[affine.For]], allow_having_children : bool):
    bands.clear()

    for loop in filter(lambda x: isinstance(x, affine.For), block.walk()):
        assert isinstance(loop, affine.For)
        n_children = get_child_loop_num(loop)

        if n_children == 0 or (n_children > 1 and allow_having_children):
            bands.append(get_loop_band_from_innermost(loop))
