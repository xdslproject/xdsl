from __future__ import annotations
from typing import Collection, Optional
from warnings import warn
import warnings
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func
from xdsl.dialects.scf import Scf
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.parser import Parser, Source
from xdsl.printer import Printer
from xdsl.rewriting.pdl.interpreter import InterpModifier, InterpResult, Interpreter, InterpreterException
from xdsl.utils import *
from xdsl.dialects.pdl import PDL, OperandOp, OperationOp, PatternOp, ReplaceOp, ResultOp, RewriteOp, TypeType, ValueType


class InterpreterWarning(Warning):
    ...


def debug(msg: str):
    warn(msg, category=InterpreterWarning)


@dataclass(frozen=True)
class LHSInterpretation(InterpModifier):
    pass


@dataclass(frozen=True)
class RHSInterpretation(InterpModifier):
    pass


@dataclass()
class PDLInterpreter(Interpreter):
    matching_env: dict[SSAValue, Attribute | SSAValue
                       | Operation] = field(default_factory=dict)
    remapping_env: dict[SSAValue,
                        Attribute | SSAValue] = field(default_factory=dict)
    native_matchers: dict[str, Callable[
        [Operation],
        Optional[Collection[Operation
                            | OpResult]]]] = field(default_factory=dict)
    native_rewriters: dict[str, Callable[
        [Operation],
        Optional[Collection[Operation
                            | OpResult]]]] = field(default_factory=dict)

    def register_native_matcher(
            self,
            matcher: Callable[[Operation],
                              Optional[Collection[Operation | OpResult]]],
            name: str):
        self.native_matchers[name] = matcher

    def register_native_rewriter(
            self,
            rewriter: Callable[[Operation],
                               Optional[Collection[Operation | OpResult]]],
            name: str):
        self.native_rewriters[name] = rewriter

    def __post_init__(self):
        self._register_all_pdl_ops()

    def _register_all_pdl_ops(self):

        self.register_interpretable_op(ModuleOp, self._interpret_module_op)
        self.register_interpretable_op(
            op_type=PatternOp, interp_fun=self._interpret_pdl_pattern_op)
        self.register_interpretable_op(
            op_type=OperationOp,
            interp_fun=self._interpret_pdl_operation_op_lhs,
            modifier=LHSInterpretation())
        self.register_interpretable_op(
            op_type=ResultOp,
            interp_fun=self._interpret_pdl_result_op_lhs,
            modifier=LHSInterpretation())
        self.register_interpretable_op(
            op_type=OperandOp,
            interp_fun=self._interpret_pdl_operand_op_lhs,
            modifier=LHSInterpretation())
        self.register_interpretable_op(
            op_type=RewriteOp,
            interp_fun=self._interpret_pdl_rewrite_op_rhs,
            modifier=RHSInterpretation())
        self.register_interpretable_op(
            op_type=OperationOp,
            interp_fun=self._interpret_pdl_operation_op_rhs,
            modifier=RHSInterpretation())
        self.register_interpretable_op(
            op_type=ReplaceOp,
            interp_fun=self._interpret_pdl_replace_op_rhs,
            modifier=RHSInterpretation())

    @staticmethod
    def _interpret_module_op(
        op: Operation,
        interpreter: Interpreter,
        *args: Any,
    ) -> InterpResult:
        if not isinstance(module_op := op, ModuleOp):
            raise InterpreterException(
                f"Got incorrect op: {op.name} in interpreter fun for ModuleOp!"
            )
        # For now just find the first pdl.PatternOp and interpret it
        for nested_op in module_op.ops:
            if isinstance(nested_op, PatternOp):
                return interpreter.interpret_op(nested_op, *args)
        return InterpResult(False, "No pdl.PatternOp found in module")

    @staticmethod
    def _interpret_pdl_pattern_op(op: Operation, interpreter: Interpreter,
                                  *args: Any) -> InterpResult:
        if not isinstance(pattern_op := op, PatternOp):
            raise InterpreterException(
                f"Got incorrect op: {op.name} in interpreter fun for PatternOp!"
            )
        # get pdl.replace op:
        rewrite_op: Optional[RewriteOp] = None
        root_op: Optional[OperationOp] = None
        for nested_op in reversed(pattern_op.body.ops):
            if isinstance(nested_op, RewriteOp):
                rewrite_op = nested_op
            elif isinstance(nested_op, OperationOp):
                root_op = nested_op
                break
        if rewrite_op is None:
            raise InterpreterException("No pdl.Rewrite found in pdl.Pattern!")
        if root_op is None:
            raise InterpreterException(
                "No valid root operation found in pattern!")

        # check pdl.ReplaceOp is well formed
        if len(rewrite_op.regions) != 1 or len(
                rewrite_op.regions[0].blocks) != 1:
            raise InterpreterException(
                "pdl.ReplaceOp must have exactly one region with one block! TODO: handle native rewrites."
            )

        # interpret matching
        if not (interp_result := interpreter.interpret_op(
                root_op, *args, modifier=LHSInterpretation())).success:
            return interp_result

        # interpret IR generation
        interpreter.interpret_op(rewrite_op, modifier=RHSInterpretation())
        # return InterpResult(True)

    @staticmethod
    def _interpret_pdl_operation_op_lhs(op: Operation,
                                        interpreter: PDLInterpreter,
                                        *args: Any) -> InterpResult:
        """
        Expects in args[0] an operation to match against
        """
        if not isinstance(operation_op := op, OperationOp):
            raise InterpreterException(
                f"Got incorrect op: {op.name} in interpreter fun for OperationOp!"
            )
        if not isinstance(payload_op := args[0], Operation):
            raise InterpreterException(f"Expected an operation as argument!")

        debug(f"matching apart: {payload_op.name}")

        # Check that name matches if given
        if operation_op.opName is not None:
            if (name_attr :=
                    operation_op.opName.data) and name_attr != payload_op.name:
                return InterpResult(
                    False,
                    f"name does not match: {name_attr} vs {payload_op.name}")

        # Check that the expected operands are present
        for idx, operand_constraint in enumerate(operation_op.operandValues):
            if not isinstance(operand_constraint, OpResult) or not (
                    isinstance(operand_constraint.op, ResultOp)
                    or isinstance(operand_constraint.op, OperandOp)):
                raise Exception(
                    f"Expected a pdl.ResultOp or pdl.OperandOp, got {operand_constraint} instead!"
                )
            if not (interp_result := interpreter.interpret_op(
                    operand_constraint.op,
                    payload_op,
                    idx,
                    modifier=LHSInterpretation())).success:
                return interp_result

        # TODO: match apart the other operands and check whether the constraints imposed by them are true
        for idx, operand_constraint in enumerate(operation_op.typeValues):
            pass

        for idx, operand_constraint in enumerate(operation_op.attributeValues):
            pass

        # record the match of this op in the matching environment so it can be referred to
        # in the rewriting part
        interpreter.matching_env[operation_op.op] = payload_op
        debug(f"sucessfully matched {operation_op.opName}")
        return InterpResult(True)

    @staticmethod
    def _interpret_pdl_operand_op_lhs(op: Operation,
                                      interpreter: PDLInterpreter,
                                      *args: Any) -> InterpResult:
        """
        Expects:
            args[0]: Operation - The Operation to expect an operand from.
            args[1]: int - The index
        """
        if not isinstance(operand_op := op, OperandOp):
            raise InterpreterException(
                f"Got incorrect op: {operand_op.name} in interpreter fun for OperandOp!"
            )
        if not isinstance(payload_op := args[0], Operation) or not isinstance(
            (idx := args[1]), int):
            raise InterpreterException(
                f"Expected an operation as argument and int as second argument!"
            )

        if len(payload_op.operands) < idx:
            return InterpResult(False,
                                "Operation has wrong number of operands.")
        # record match in environment
        interpreter.matching_env[operand_op.value] = payload_op.operands[idx]
        return InterpResult(True)

    @staticmethod
    def _interpret_pdl_result_op_lhs(op: Operation,
                                     interpreter: PDLInterpreter,
                                     *args: Any) -> InterpResult:
        """
        Expects:
            args[0]: Operation - The Operation to get the result from.
            args[1]: int - The index 
        """

        if not isinstance(result_op := op, ResultOp):
            raise InterpreterException(
                f"Got incorrect op: {op.name} in interpreter fun for ResultOp!"
            )
        if not isinstance(result_op.parent_, OpResult):
            raise InterpreterException(f"TODO")
        if not isinstance(payload_op := args[0], Operation) or not isinstance(
            (idx := args[1]), int):
            raise InterpreterException(
                f"Expected an operation as argument and int as second argument!"
            )
        if len(payload_op.operands) < idx:
            return InterpResult(False,
                                "Operation has wrong number of operands.")
        if not isinstance(
            (payload_operand := payload_op.operands[idx]), OpResult):
            return InterpResult(False,
                                f"Operand is not the result of an Operation!")

        if len(payload_operand.op.results) < (result_index :=
                                              result_op.index.value.data):
            return InterpResult(
                False, f"Result index of {result_index} out of bounds. \
                Op: {payload_operand.op.name} only has {len(payload_operand.op.results)} results!"
            )
        # record match in environment
        interpreter.matching_env[
            result_op.val] = payload_operand.op.results[result_index]
        return interpreter.interpret_op(result_op.parent_.op,
                                        payload_operand.op,
                                        modifier=LHSInterpretation())

    @staticmethod
    def _interpret_pdl_rewrite_op_rhs(op: Operation,
                                      interpreter: PDLInterpreter,
                                      *args: Any) -> InterpResult:
        if not isinstance(rewrite_op := op, RewriteOp):
            raise InterpreterException(
                f"Got incorrect op: {op.name} in interpreter fun for RewriteOp!"
            )

        if rewrite_op.body is None:
            raise InterpreterException(
                f"For now we only support RewriteOps with a Region!")

        for nested_op in rewrite_op.body.ops:
            if not (interp_result := interpreter.interpret_op(
                    nested_op, modifier=RHSInterpretation())).success:
                return interp_result

        return InterpResult(True)

    @staticmethod
    def _interpret_pdl_operation_op_rhs(op: Operation,
                                        interpreter: PDLInterpreter,
                                        *args: Any) -> InterpResult:
        if not isinstance(operation_op := op, OperationOp):
            raise InterpreterException(
                f"Got incorrect op: {op.name} in interpreter fun for OperationOp!"
            )

        # TODO: generate new operation

        return InterpResult(True)

    @staticmethod
    def _interpret_pdl_replace_op_rhs(op: Operation,
                                      interpreter: PDLInterpreter,
                                      *args: Any) -> InterpResult:
        if not isinstance(replace_op := op, ReplaceOp):
            raise InterpreterException(
                f"Got incorrect op: {op.name} in interpreter fun for ReplaceOp!"
            )

        # TODO: handle replaceOp

        return InterpResult(True)


if __name__ == "__main__":
    IR = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 4 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.constant() ["value" = 1 : !i32]
  %3 : !i32 = arith.addi(%2 : !i32, %1 : !i32)
  %4 : !i32 = arith.addi(%3 : !i32, %0 : !i32)
  func.return(%4 : !i32)
}
"""

    # The rewrite below matches the second addition as root op

    rewrites = """"builtin.module"() ({
  "pdl.pattern"() ({
      %0 = "pdl.operand"() : () -> !pdl.value
      %1 = "pdl.operand"() : () -> !pdl.value
      %2 = "pdl.type"() : () -> !pdl.type
      %3 = "pdl.operation"(%0, %1, %2) {attributeValueNames = [], opName = "arith.addi", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
      %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      %5 = "pdl.operand"() : () -> !pdl.value
      %6 = "pdl.operation"(%4, %5, %2) {attributeValueNames = [], opName = "arith.addi", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
      "pdl.rewrite"(%6) ({
        %7 = "pdl.operation"(%5, %4, %2) {attributeValueNames = [], opName = "arith.addi", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
        "pdl.replace"(%6, %7) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
      }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
    }) {benefit = 2 : i16} : () -> ()
}) : () -> ()
"""

    warnings.simplefilter("always", category=InterpreterWarning)

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Scf)
    ctx.register_dialect(PDL)

    ir_parser = Parser(ctx, IR)
    ir_module: Operation = ir_parser.parse_op()

    printer = Printer(target=Printer.Target.MLIR)
    printer.print_op(ir_module)

    pdl_parser = Parser(ctx, rewrites, source=Source.MLIR)
    pdl_module: Operation = pdl_parser.parse_op()

    pdl_interpreter = PDLInterpreter()
    assert isinstance(ir_module, ModuleOp)

    # We don't yet have a way to traverse to program.
    # In the old interpreter we did that using elevate.
    # So for now we just try to apply our rewrite everywhere.
    for op in ir_module.ops:
        result = pdl_interpreter.interpret_op(pdl_module, op)
        if result and not result.success:
            debug(result.error_msg)
        else:
            break
