from xdsl.dialects import memref, scf
from xdsl.ir import Block, Operation, SSAValue


# TODO replace by functionality (when added) as described in https://github.com/xdslproject/xdsl/issues/2128
def get_operation_at_index(block: Block, idx: int) -> Operation:
    """Get an operation by its position in its parent block."""

    for _idx, block_op in enumerate(block.ops):
        if idx == _idx:
            return block_op

    raise ValueError(
        f"Cannot get operation by out-of-bounds index {idx} in its parent block."
    )


def find_corresponding_store(load: memref.Load):
    parent_block = load.parent_block()

    if parent_block is None:
        return None

    found_op = None

    for op in parent_block.ops:
        if (
            isinstance(op, memref.Store)
            and op.memref == load.memref
            and op.indices == load.indices
        ):
            if found_op is None:
                found_op = op
            else:
                return None

    return found_op


def is_loop_dependent(val: SSAValue, loop: scf.For):
    worklist: set[SSAValue] = set()
    visited: set[SSAValue] = set()

    worklist.add(val)

    while worklist:
        val = worklist.pop()
        if val in visited:
            continue

        visited.add(val)

        if val is loop.body.block.args[0]:
            return True

        if isinstance(val.owner, Operation):
            for oprnd in val.owner.operands:
                if oprnd not in visited:
                    worklist.add(oprnd)
        else:
            for arg in val.owner.args:
                if arg not in visited:
                    worklist.add(arg)

    return False
