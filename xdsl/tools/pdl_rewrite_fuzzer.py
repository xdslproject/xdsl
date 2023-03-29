import argparse
from dataclasses import dataclass, field
from random import randrange

from xdsl.ir import Operation, Region, SSAValue
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl.dialects.pdl import OperandOp, OperationOp, PatternOp, ResultOp, TypeOp, TypeType, ValueType
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


@dataclass
class FuzzerContext():
    values: list[SSAValue] = field(default_factory=list)
    operations: list[OperationOp] = field(default_factory=list)


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

    op_name = FuzzerOptions.available_operations[randrange(
        0, len(FuzzerOptions.available_operations))]

    op = OperationOp.get(StringAttr(op_name), None, operands, None, results)
    new_ops.append(op)
    ctx.operations.append(op)

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

    # Generate a the matching part
    matched_ops: list[Operation] = []
    for _ in range(num_matched_operations):
        matched_ops.extend(generate_random_matched_operation(ctx))

    body = Region.from_operation_list(matched_ops)
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