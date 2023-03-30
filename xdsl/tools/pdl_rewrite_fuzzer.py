import argparse
from dataclasses import dataclass, field
from random import randrange

from xdsl.ir import Operation, Region, SSAValue
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl.dialects.pdl import EraseOp, OperandOp, OperationOp, PatternOp, ReplaceOp, ResultOp, RewriteOp, TypeOp, TypeType, ValueType
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp, StringAttr, i32

i16 = IntegerType(16)


class FuzzerOptions:
    min_operands = 0
    max_operands = 2
    min_results = 0
    max_results = 2
    min_match_operations = 1
    max_match_operations = 4
    min_rewrite_operations = 1
    max_rewrite_operations = 3
    available_operations = ["test.op1", "test.op2", "test.op3"]

    @staticmethod
    def get_random_operation_name():
        return FuzzerOptions.available_operations[randrange(
            0, len(FuzzerOptions.available_operations))]


@dataclass
class FuzzerContext():
    values: list[SSAValue] = field(default_factory=list)
    operations: list[OperationOp] = field(default_factory=list)

    def get_random_value(self) -> SSAValue:
        assert len(self.values) != 0
        return self.values[randrange(0, len(self.values))]

    def get_random_operation(self) -> OperationOp:
        assert len(self.operations) != 0
        return self.operations[randrange(0, len(self.operations))]


def generate_random_operand(
        ctx: FuzzerContext) -> tuple[SSAValue, list[Operation]]:
    """
    Generate a random operand.
    It is either a new `pdl.operand`, or an existing one in the context.
    """
    if len(ctx.values) != 0 and randrange(0, 2) == 0:
        return ctx.values[randrange(0, len(ctx.values))], []
    new_type = TypeOp.create(result_types=[TypeType()],
                             attributes={"constantType": i32})
    new_operand = OperandOp.create(result_types=[ValueType()],
                                   operands=[new_type.result])
    return new_operand.value, [new_type, new_operand]


def generate_random_matched_operation(ctx: FuzzerContext) -> list[Operation]:
    """
    Generate a random `pdl.operation`, along with new
    `pdl.operand` and `pdl.type` if necessary.
    """
    num_operands = randrange(FuzzerOptions.min_operands,
                             FuzzerOptions.max_operands + 1)
    num_results = randrange(FuzzerOptions.min_results,
                            FuzzerOptions.max_results + 1)
    new_ops: list[Operation] = []

    operands: list[SSAValue] = []
    results: list[SSAValue] = []
    for _ in range(num_operands):
        operand, operand_ops = generate_random_operand(ctx)
        operands.append(operand)
        new_ops.extend(operand_ops)

    for _ in range(num_results):
        new_type = TypeOp.create(result_types=[TypeType()],
                                 attributes={"constantType": i32})
        results.append(new_type.result)
        new_ops.extend([new_type])

    op_name = FuzzerOptions.get_random_operation_name()

    op = OperationOp.get(StringAttr(op_name), None, operands, None, results)
    new_ops.append(op)
    ctx.operations.append(op)

    for result_idx in range(num_results):
        result = ResultOp.get(IntegerAttr[IntegerType](result_idx, i32), op.op)
        new_ops.append(result)
        ctx.values.append(result.val)
    return new_ops


def generate_random_rewrite_operation(ctx: FuzzerContext) -> list[Operation]:
    """
    Generate a random operation in the rewrite part of the pattern.
    This can be either an `operation`, an `erase`, or a `replace`.
    """
    operation_choice = randrange(0, 4)

    # Erase operation
    if operation_choice == 0:
        op = ctx.get_random_operation()
        return [EraseOp.get(op.op)]

    # Replace operation with another operation
    if operation_choice == 1:
        op = ctx.get_random_operation()
        op2 = ctx.get_random_operation()
        return [ReplaceOp.get(op.op, op2.op)]

    # Replace operation with multiple values
    if operation_choice == 2:
        op = ctx.get_random_operation()
        # If we need values but we don't have, we restart
        if len(op.results) != 0 and len(ctx.values) == 0:
            return generate_random_rewrite_operation(ctx)
        values = [ctx.get_random_value() for _ in op.results]
        return [ReplaceOp.get(op.op, None, values)]

    # Create a new operation
    assert operation_choice == 3
    op_name = FuzzerOptions.get_random_operation_name()
    num_operands = randrange(FuzzerOptions.min_operands,
                             FuzzerOptions.max_operands + 1)
    num_results = randrange(FuzzerOptions.min_results,
                            FuzzerOptions.max_results + 1)

    # If we need values but we don't have, we restart
    if num_operands != 0 and len(ctx.values) == 0:
        return generate_random_rewrite_operation(ctx)

    new_ops: list[Operation] = []
    operands = [ctx.get_random_value() for _ in range(num_operands)]
    results: list[SSAValue] = []
    for _ in range(num_results):
        new_type = TypeOp.create(result_types=[TypeType()],
                                 attributes={"constantType": i32})
        results.append(new_type.result)
        new_ops.extend([new_type])

    op = OperationOp.get(StringAttr(op_name), None, operands, None, results)
    ctx.operations.append(op)
    new_ops.append(op)

    for result_idx in range(num_results):
        result = ResultOp.get(IntegerAttr[IntegerType](result_idx, i32), op.op)
        new_ops.append(result)
        ctx.values.append(result.val)

    return new_ops


def generate_random_pdl_rewrite() -> PatternOp:
    """
    Generate a random match part of a `pdl.rewrite`.
    """
    ctx = FuzzerContext()
    num_matched_operations = randrange(FuzzerOptions.min_match_operations,
                                       FuzzerOptions.max_match_operations + 1)
    num_rewrite_operations = randrange(
        FuzzerOptions.min_rewrite_operations,
        FuzzerOptions.max_rewrite_operations + 1)

    # Generate a the matching part
    matched_ops: list[Operation] = []
    for _ in range(num_matched_operations):
        matched_ops.extend(generate_random_matched_operation(ctx))

    # Get the last operation in the match, this is the one we use to rewrite
    rewritten_op = ctx.operations[-1]

    # Generate the rewrite part
    rewrite_ops: list[Operation] = []
    for _ in range(num_rewrite_operations):
        rewrite_ops.extend(generate_random_rewrite_operation(ctx))

    region = Region.from_operation_list(rewrite_ops)

    rewrite = RewriteOp.get(None, rewritten_op.op, [], region)

    body = Region.from_operation_list(matched_ops + [rewrite])
    return PatternOp.get(IntegerAttr[IntegerType](0, i16), None, body)


class PDLRewriteFuzzMain(xDSLOptMain):

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)
        pass

    def run(self):
        pattern = generate_random_pdl_rewrite()
        module = ModuleOp.from_region_or_ops([pattern])
        contents = self.output_resulting_program(module)
        self.print_to_output_stream(contents)


if __name__ == "__main__":
    PDLRewriteFuzzMain().run()