from xdsl.dialects.experimental.hida_functional import TaskOp, DispatchOp, YieldOp
from xdsl.irdl import Operation, Block
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.traits import IsTerminator
from xdsl.utils.hints import isa
from xdsl.rewriter import InsertPoint

def dispatch_block(block : Block):
    #assert isinstance(block.parent_region(), Region)

    dispatch_block_ops : list[Operation] = []
    for op in block.ops:
        # The original terminator of the parent operation is kept in the parent block
        if not op.has_trait(IsTerminator):
            op.detach()
            dispatch_block_ops.append(op)

    dispatch_block = Block(dispatch_block_ops)
    dispatch_op = DispatchOp(dispatch_block)

    return dispatch_op


def fuse_ops_into_task(ops : list[Operation], rewriter: PatternRewriter, insert_to_last_op : bool):
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
    list(map(lambda x: task.region.block.add_op(x), ops))
    task.region.block.add_op(yield_op)

    # Propagate output values of the task to users of the outputs of the member operations
    for res_idx,res in enumerate(task.results):
        output_values[res_idx].replace_by_if(res, lambda use: not task.is_ancestor(use.operation))

    # Inline all sub tasks
    for subtask in filter(lambda x: isinstance(x, TaskOp), ops):
        assert isinstance(subtask, TaskOp) and isinstance(subtask.region.block.last_op, Operation)

        yield_operands = list(map(lambda x: x.owner, subtask.region.block.last_op.operands))
        assert isa(yield_operands, list[Operation])
        list(map(lambda x: x.detach(), yield_operands))

        rewriter.replace_op(subtask, yield_operands, new_results=[res for yield_operand in yield_operands for res in yield_operand.results])
