import argparse
from dataclasses import dataclass, field
from random import randrange

from xdsl.ir import Operation, SSAValue
from xdsl_opt_main import xDSLOptMain

from xdsl.dialects.pdl import OperandOp, OperationOp, TypeOp, TypeType, ValueType
from xdsl.dialects.builtin import i32

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

def generate_random_operand(ctx: FuzzerContext) -> tuple[SSAValue, list[Operation]]:
    """
    Generate a random operand.
    It is either a new `pdl.operand`, or an existing one in the context.
    """
    if len(ctx.values) != 0 and randrange(0, 2) == 0:
        return ctx.values[randrange(0, len(ctx.values))], []
    new_type = TypeOp.create(result_types=[TypeType()], attributes={"constantType": i32})
    new_operand = OperandOp.create(result_types=[ValueType()], operands=[new_type.result])
    return new_operand.value, [new_type, new_operand]

def generate_random_operation(ctx: FuzzerContext) -> tuple[Operation, list[Operation]]:
    """
    Generate a random `pdl.operation`, along with new
    `pdl.operand` and `pdl.type` if necessary.
    """
    num_operands = randrange(FuzzerOptions.min_operands, FuzzerOptions.max_operands + 1)
    num_results = randrange(FuzzerOptions.min_results, FuzzerOptions.max_results + 1)
    new_ops: list[Operation] = []

    operands: list[SSAValue] = []
    results: list[SSAValue] = []
    for _ in range(num_operands):
        operand, operand_ops = generate_random_operand(ctx)
        operands.append(operand)
        new_ops.extend(operand_ops)
    
    for _ in range(num_results):
        new_type = TypeOp.create(result_types=[TypeType()], attributes={"constantType": i32})
        results.append(new_type.result)
        new_ops.extend([new_type])

    op = 


def generate_random_pdl_matcher():
    


class PDLRewriteFuzzMain(xDSLOptMain):

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)
        arg_parser.add_argument("--mlir-executable", type=str, required=True)

    def run(self):
        module = self.parse_input()
        for _ in range(0, 1000):
            fuzz_pdl_matches(module, self.ctx, self.args.mlir_executable)


if __name__ == "__main__":
    PDLRewriteFuzzMain().run()