from dataclasses import dataclass, field
from typing import IO, cast

from typing_extensions import Self

from xdsl.dialects import builtin, func
from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue
from xdsl.printer import Printer


class BlockInfoBuilder:
    def __init__(self, block: Block):
        self.out_values: set[SSAValue] = set()
        self.def_values: set[SSAValue] = set()
        self.use_values: set[SSAValue] = set()
        self.in_values: set[SSAValue] = set()
        self.block = block

        def gather_out_values(value: SSAValue):
            # Check whether this value will be in the outValues set (its uses escape
            # this block). Due to the SSA properties of the program, the uses must
            # occur after the definition. Therefore, we do not have to check
            # additional conditions to detect an escaping value.
            for use_op in [use.operation for use in value.uses]:
                owner_block = use_op.parent_block()
                # Find an owner block in the current region. Note that a value does not
                # escape this block if it is used in a nested region.
                parent_region = block.parent_region()
                assert isinstance(parent_region, Region)
                assert isinstance(owner_block, Block)
                owner_block = parent_region.find_ancestor_block_in_region(owner_block)
                assert owner_block
                assert "Use leaves the current parent region"
                if owner_block != block:
                    self.out_values.add(value)
                    break

        # Mark all block arguments (phis) as defined
        for argument in block.args:
            # Insert value into the set of defined values
            self.def_values.add(argument)

            # Gather all out values of all arguments in the current block.
            gather_out_values(argument)

        # Gather out values of all operations in the current block.
        for operation in block.ops:
            for result in operation.results:
                gather_out_values(result)

        # Mark all nested operation results as defined, and nested operation
        # operands as used. All defined value will be removed from the used set
        # at the end.
        for op in block.walk():
            for result in op.results:
                self.def_values.add(result)
            for operand in op.operands:
                self.use_values.add(operand)
            for region in op.regions:
                for child in region.blocks:
                    for arg in child.args:
                        self.def_values.add(arg)
        self.use_values = self.use_values.difference(self.def_values)

    # Updates live-in information of the current block. To do so it uses the
    # default liveness-computation formula: newIn = use union out \ def. The
    # methods returns true, if the set has changed (newIn != in), false
    # otherwise.
    def update_livein(self):
        new_in = self.use_values
        new_in = new_in.union(self.out_values)
        new_in = new_in.difference(self.def_values)

        # It is sufficient to check the set sizes (instead of their contents) since
        # the live-in set can only grow monotonically during all update operations.
        if len(new_in) == len(self.in_values):
            return False

        self.in_values = new_in.copy()
        return True

    # Updates live-out information of the current block. It iterates over all
    # successors and unifies their live-in values with the current live-out
    # values.
    def update_liveout(self, builders: dict[Block, Self]):
        assert self.block.last_op
        for succ in self.block.last_op.successors:
            builder = builders[succ]
            self.out_values = self.out_values.union(builder.in_values)


# Builds the internal liveness block mapping.
def build_block_mapping(operation: Operation) -> dict[Block, BlockInfoBuilder]:
    to_process: set[Block] = set()
    builders: dict[Block, BlockInfoBuilder] = dict()

    # for op in operation.walk():
    #    for block in [block for region in op.regions for block in region.blocks]:
    for block in operation.walk_blocks():
        assert isinstance(block, Block)
        if block not in builders:
            builders[block] = BlockInfoBuilder(block)

        builder = builders[block]

        if builder.update_livein():
            list(
                map(
                    lambda x: to_process.add(x), [pred for pred in block.predecessors()]
                )
            )

    # Propagate the in and out-value sets (fixpoint iteration).
    while to_process:
        current = to_process.pop()
        builder = builders[current]

        # Update the current out values.
        builder.update_liveout(builders)

        # Compute (potentially) updated live in values.
        if builder.update_livein():
            list(
                map(
                    lambda x: to_process.add(x),
                    [pred for pred in current.predecessors()],
                )
            )

    return builders


# ===----------------------------------------------------------------------===//
# LivenessBlockInfo
# ===----------------------------------------------------------------------===//


# This class represents liveness information on block level.
@dataclass
class LivenessBlockInfo:
    in_values: set[SSAValue] = field(default_factory=set[SSAValue])
    out_values: set[SSAValue] = field(default_factory=set[SSAValue])

    def __init__(self, block: Block):
        self.block = block

    # Returns True if the given value is in the live-in set.
    def is_livein(self, value: SSAValue):
        return value in self.in_values

    # Returns True if the given vlaue is in the live-out set.
    def is_liveout(self, value: SSAValue):
        return value in self.out_values

    # Gets the start operation for the given value (must be referenced in this block).
    def get_start_operation(self, value: SSAValue):
        defining_op = value.owner if isinstance(value.owner, Operation) else None
        # The given value is either live-in or is defined in the scope of this block
        if self.is_livein(value) or not defining_op:
            return self.block.first_op

        return defining_op

    # Gets the end operation for the given value using the start operation provided (
    # must be referenced in this block)
    def get_end_operation(self, value: SSAValue, start_operation: Operation):
        # The given value is either dying in this block or live-out.
        if self.is_liveout(value):
            return self.block.last_op

        # Resolve the last operation (must exist by definition).
        end_operation = start_operation
        for use_op in [use.operation for use in value.uses]:
            use_op = self.block.find_ancestor_op_in_block(use_op)
            # Check whether the use is in our block and after the current end operation.
            if use_op and end_operation.is_before_in_block(use_op):
                end_operation = use_op

        return end_operation

    # Return the values that are currently live as of the given operation.
    def currently_live_values(self, op: Operation, output: IO[str]):
        live_set: set[SSAValue] = set()

        # Given a value, check which ops are within its live range. For each of
        # those ops, add the value to the set of live values as-of that op
        def add_value_to_currently_live_sets(value: SSAValue):
            start_of_live_range = (
                value.owner if isinstance(value.owner, Operation) else None
            )
            end_of_live_range = None

            # If it's a live in or a block argument, then the start is the beginning of
            # the block.
            if self.is_livein(value) or isinstance(value, BlockArgument):
                start_of_live_range = self.block.first_op
            else:
                assert isinstance(start_of_live_range, Operation)
                start_of_live_range = self.block.find_ancestor_op_in_block(
                    start_of_live_range
                )

            # If it's a live out, then the end is the back of the block.
            if self.is_liveout(value):
                end_of_live_range = self.block.last_op

            # We must have at least a start_of_live_range at this point. Given this, we can
            # use the existing get_end_operation to find the end of the live range.
            if start_of_live_range and not end_of_live_range:
                end_of_live_range = self.get_end_operation(value, start_of_live_range)

            assert end_of_live_range
            assert "Must have end_of_live_range at this point!"
            # If this op is within the live range, insert the value into the set.
            assert isinstance(start_of_live_range, Operation)
            if not (
                op.is_before_in_block(start_of_live_range)
                or end_of_live_range.is_before_in_block(op)
            ):
                live_set.add(value)

        # Handle block arguments if any.
        for arg in self.block.args:
            add_value_to_currently_live_sets(arg)

        # Handle live-ins. Between the live ins and all the op results that gives us every value
        # in the block.
        for in_val in self.in_values:
            add_value_to_currently_live_sets(in_val)

        # Now walk the block and handle all the values used in the block and values defined by the
        # block.
        for _op in self.block.ops:
            for result in _op.results:
                add_value_to_currently_live_sets(result)

        return live_set


# ===----------------------------------------------------------------------===//
# Liveness
# ===----------------------------------------------------------------------===//
block_mapping: dict[Block, LivenessBlockInfo] = dict()


# Creates a new Liveness analysis that computes liveness information for all
# associated regions.
@dataclass
class Liveness:
    operation: Operation

    def __init__(self, op: Operation):
        self.operation = op
        self.build(op)

    # Initializes the internal mappings
    def build(self, op: Operation):
        # Build internal block mapping
        builders: dict[Block, BlockInfoBuilder] = dict()
        builders = build_block_mapping(op)

        # Store internal block data
        for block in builders:
            builder = builders[block]
            block_mapping[block] = LivenessBlockInfo(block)

            block_mapping[block].block = builder.block
            block_mapping[block].in_values = builder.in_values.copy()
            block_mapping[block].out_values = builder.out_values.copy()

    # Gets liveness info (if any) for the given value.
    def resolve_liveness(self, value: SSAValue) -> list[Operation]:
        to_process: list[Block] = []
        visited: set[Block] = set()
        result: list[Operation] = []

        # Start with the defining block
        if isinstance(def_op := value.owner, Operation):
            current_block = def_op.parent_block()
        else:
            assert isinstance(value, Block)
            current_block = cast(BlockArgument, value).owner

        assert isinstance(current_block, Block)
        to_process.append(current_block)
        visited.add(current_block)

        # Start with all associated blocks.
        for use in value.uses:
            use_block = use.operation.parent_block()
            assert isinstance(use_block, Block)
            if use_block not in visited:
                to_process.append(use_block)
                visited.add(use_block)
        while to_process:
            # Get block and block liveness information
            block = to_process[-1]
            to_process.pop()
            block_info = self.get_liveness(block)

            if not block_info:
                continue

            # Note that start and end will be in the same block.
            start = block_info.get_start_operation(value)
            assert isinstance(start, Operation)
            end = block_info.get_end_operation(value, start)

            assert start
            result.append(start)
            while start != end:
                start = start.next_op
                assert isinstance(start, Operation)
                result.append(start)

            assert block.last_op
            for successor in block.last_op.successors:
                if (succ_liveness := self.get_liveness(successor)) is None:
                    continue
                if succ_liveness.is_livein(value) and successor not in visited:
                    to_process.append(successor)
                    visited.add(successor)
        return result

    # Gets liveness info (if any) for the block.
    def get_liveness(self, block: Block):
        try:
            it = block_mapping[block]
            return it
        except KeyError:
            return None

    # Returns a reference to a set containing live-in values.
    def get_livein(self, block: Block):
        if (_liveness := self.get_liveness(block)) is None:
            return None
        else:
            return _liveness.in_values

    # Returns a reference to a set containing live-out values.
    def get_liveout(self, block: Block):
        if (_liveness := self.get_liveness(block)) is None:
            return None
        else:
            return _liveness.out_values

    # Returns true if `value` is not live after `operation`.
    def is_dead_after(self, value: SSAValue, operation: Operation):
        block = operation.parent_block()
        assert isinstance(block, Block)
        block_info = self.get_liveness(block)
        assert block_info

        # The given value escapes the associated block.
        if block_info.is_liveout(value):
            return False

        end_operation = block_info.get_end_operation(value, operation)
        assert isinstance(end_operation, Operation)
        # If the operation is a real user of `value` the first check is sufficient.
        # If not, we will have to test whether the end operation is executed before
        # the given operation in the block.
        return end_operation == operation or end_operation.is_before_in_block(operation)

    # Dumps the liveness information to the given stream.
    def print(self, output: IO[str], printer: Printer):
        print("// ---- Liveness ----", file=output)

        # Builds unique block/value mappings for testing purposes.
        block_ids: dict[Block, int] = dict()
        operation_ids: dict[Operation, int] = dict()
        value_ids: dict[SSAValue, int] = dict()

        for block in self.operation.walk_blocks():
            assert isinstance(block, Block)
            block_ids[block] = len(block_ids)

            for argument in block.args:
                value_ids[argument] = len(value_ids)

            for operation in block.ops:
                operation_ids[operation] = len(operation_ids)
                for result in operation.results:
                    value_ids[result] = len(value_ids)

        # Local printing helpers
        def print_value_ref(value: SSAValue):
            if isinstance(value.owner, Operation):
                print(f" val_{value_ids[value]}", file=output, end="")
            else:
                block_arg = cast(BlockArgument, value)
                print(
                    f"arg{block_arg.index}@{block_ids[block_arg.owner]}",
                    file=output,
                    end="",
                )

            print(" ", file=output, end="")

        def print_value_refs(values: set[SSAValue]):
            ordered_values: list[SSAValue] = list(values)

            ordered_values.sort(key=lambda x: value_ids[x])
            for value in ordered_values:
                print_value_ref(value)

        # Dump information about in and out values.
        for block in self.operation.walk_blocks():
            assert isinstance(block, Block)
            print(f"// - Block: {block_ids[block]}", file=output)
            liveness = self.get_liveness(block)
            assert liveness
            print("// --- LiveIn: ", file=output, end="")
            print_value_refs(liveness.in_values)
            print("\n// --- LiveOut: ", file=output, end="")
            print_value_refs(liveness.out_values)
            print("\n", file=output, end="")

            # Print liveness intervals.
            print("// --- BeginLivenessIntervals", file=output, end="")
            for op in block.ops:
                if len(op.results) < 1:
                    continue
                print("", file=output)
                for result in op.results:
                    print("//", file=output, end="")
                    print_value_ref(result)
                    print(":", file=output, end="")
                    live_operations = self.resolve_liveness(result)
                    live_operations.sort(key=lambda x: operation_ids[x])

                    for operation in live_operations:
                        print("\n//     ", file=output, end="")
                        printer.print_op(operation)

            print("\n// --- EndLivenessIntervals", file=output)

            # Print currently live values.
            print("// --- BeginCurrentlyLive", file=output)
            for op in block.ops:
                currently_live = liveness.currently_live_values(op, output)
                if not currently_live:
                    continue
                print("//     ", file=output, end="")
                printer.print_op(op)
                print(" [", file=output, end="")
                print_value_refs(currently_live)
                print("\b]\n", file=output, end="")

            print("// --- EndCurrentlyLive", file=output)

        print("// -------------------", file=output)


def print_liveness(program: builtin.ModuleOp, output: IO[str]):
    printer = Printer(
        stream=output,
    )

    for func_op in filter(lambda x: isinstance(x, func.FuncOp), program.walk()):
        liveness = Liveness(func_op)
        liveness.print(output, printer)
