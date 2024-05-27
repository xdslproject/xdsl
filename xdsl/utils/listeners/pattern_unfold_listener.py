from collections.abc import Sequence

from xdsl.ir import Block, BlockArgument, Operation, OpResult, SSAValue
from xdsl.pattern_rewriter import PatternRewriterListener
from xdsl.utils.diagnostic import Diagnostic


def print_operation_insertion(op: Operation):
    d = Diagnostic()
    d.add_message(op, "Operation inserted")
    print(d.get_output(op))


def print_operation_removal(op: Operation):
    d = Diagnostic()
    d.add_message(op, "Removing this operation")
    print(d.get_output(op))


def print_operation_replacement(op: Operation, new_results: Sequence[SSAValue | None]):
    d = Diagnostic()
    if len(op.results) == len(new_results):
        for new_r, old_r in zip(new_results, op.results):
            if new_r is not None:
                if isinstance(new_r, OpResult):
                    owner = new_r.owner
                    d.add_message(
                        owner,
                        f"Result #{new_r.index} is replacing result #{old_r.index}.",
                    )
                elif isinstance(new_r, BlockArgument):
                    block = new_r.owner
                    region = block.parent
                    if region is None:
                        continue
                    owner = region.parent
                    if owner is None:
                        continue
                    d.add_message(
                        owner,
                        f"Argument #{new_r.index} of block #{region.get_block_index(block)} is replacing result #{old_r.index}.",
                    )

    d.add_message(op, "Replacing this operation")

    print(d.get_output(op))


def print_operation_modification(op: Operation):
    d = Diagnostic()
    d.add_message(op, "Modifying this operation")
    print(d.get_output(op))


def print_block_creation(b: Block):
    d = Diagnostic()
    op = b.parent_op()
    if op is not None:
        d.add_message(op, "Block created")
    print(d.get_output(b))


class PatternUnfoldListener(PatternRewriterListener):

    def __init__(
        self,
        *,
        insertion: bool = True,
        removal: bool = True,
        modification: bool = True,
        replacement: bool = True,
        block_creation: bool = True,
    ):
        operation_insertion_handler = [print_operation_insertion] if insertion else []
        operation_removal_handler = [print_operation_removal] if removal else []
        operation_modification_handler = (
            [print_operation_modification] if modification else []
        )
        operation_replacement_handler = (
            [print_operation_replacement] if replacement else []
        )
        block_creation_handler = [print_block_creation] if block_creation else []
        super().__init__(
            operation_insertion_handler=operation_insertion_handler,
            block_creation_handler=block_creation_handler,
            operation_removal_handler=operation_removal_handler,
            operation_modification_handler=operation_modification_handler,
            operation_replacement_handler=operation_replacement_handler,
        )
