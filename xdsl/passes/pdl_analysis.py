from __future__ import annotations
from dataclasses import dataclass, field
from typing import Type
import warnings
from warnings import warn
from xdsl.dialects import pdl
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.dialects.cf import Cf
from xdsl.dialects.cmath import CMath
from xdsl.dialects.func import Func
from xdsl.dialects.gpu import GPU
from xdsl.dialects.irdl import IRDL
from xdsl.dialects.llvm import LLVM
from xdsl.dialects.memref import MemRef
from xdsl.dialects.mpi import MPI
from xdsl.dialects.scf import Scf
from xdsl.dialects.vector import Vector
from xdsl.ir import MLContext, OpResult, Operation, SSAValue, TerminatorOp
from xdsl.printer import Printer


class PDLDebugWarning(Warning):
    ...


enable_prints = True
warnings.simplefilter("ignore", category=PDLDebugWarning)


def debug(msg: str):
    warn(msg)


@dataclass()
class UseInfo():
    dominated_by: list[AnalyzedPDLOperation] = field(default_factory=list)


@dataclass
class AnalyzedPDLOperation:
    """
    This class bundles a pdl.OperationOp with information from the
    analysis of the pdl.PatternOp that contains it.
    """
    pdl_op: pdl.OperationOp
    op_type: Type[Operation]
    matched: bool = True
    erased_by: pdl.EraseOp | None = None
    replaced_by: list[SSAValue] | AnalyzedPDLOperation | None = None
    use_info: UseInfo = UseInfo()

    @property
    def is_terminator(self):
        return issubclass(self.op_type, TerminatorOp)


@dataclass(init=False)
class PDLAnalysis:
    """
    This class is responsible for analyzing a given PDL pattern, 
    gathering information about the matching part pattern, 
    and the right-hand side of the pattern. Information about the 
    pattern at hand is collected at initialization. It stores the analyzed 
    operations in the `analyzed_ops` attribute, and provides properties
    to access different sets of analyzed operations such as `matched_ops`,
    `erased_ops`, `terminator_matches`, and `generated_ops`. 
    """

    context: MLContext
    pattern_op: pdl.PatternOp
    root_op: pdl.OperationOp
    rewrite_op: pdl.RewriteOp
    analyzed_ops: list[AnalyzedPDLOperation] = field(default_factory=list)
    visited_ops: set[Operation] = field(default_factory=set)

    @property
    def matched_ops(self):
        return [op for op in self.analyzed_ops if op.matched]

    @property
    def erased_ops(self):
        return [op for op in self.analyzed_ops if op.erased_by]

    @property
    def terminator_matches(self):
        return [
            op for op in self.analyzed_ops
            if (op.is_terminator or op.op_type == Operation) and op.matched
        ]

    @property
    def generated_ops(self):
        return [op for op in self.analyzed_ops if not op.matched]

    def __init__(self, context: MLContext, pattern_op: pdl.PatternOp):
        self.context = context
        self.pattern_op = pattern_op
        self.analyzed_ops = []
        self.visited_ops = set()

        if isinstance(pattern_op.body.ops[-1], pdl.RewriteOp):
            self.rewrite_op: pdl.RewriteOp = pattern_op.body.ops[-1]
        else:
            raise Exception(
                "pdl.PatternOp must have a pdl.RewriteOp as terminator!")

        if isinstance(pattern_op.body.ops[-2], pdl.OperationOp):
            self.root_op: pdl.OperationOp = pattern_op.body.ops[-2]
        else:
            raise Exception(
                "pdl.PatternOp must have a pdl.OperationOp as root!")

        # Gather information about the matching part pattern
        self._trace_matching_op(self.root_op)
        self.visited_ops.add(self.rewrite_op)
        # Check whether all ops in the pattern are reachable
        if len(pattern_op.body.ops) != len(self.visited_ops):
            if enable_prints:
                debug(
                    f"{abs(len(pattern_op.body.ops) - len(self.visited_ops))} Unreachable PDL ops in pattern!"
                )
                debug("Full pattern:")
                printer = Printer()
                printer.print_op(pattern_op)
                debug("Unreachable ops:")
                for pdl_op in pattern_op.body.ops:
                    if pdl_op not in self.visited_ops:
                        printer.print_op(pdl_op)

        # Gather information about the rhs of the pattern
        self._analyze_pdl_rewrite_op(self.rewrite_op)

    def get_analysis_for_pdl_op(
            self, op: pdl.OperationOp) -> AnalyzedPDLOperation | None:
        """
        If `op` is already part of the analysis, return the corresponding
        AnalyzedPDLOperation that contains the analysis information.
        """
        for analyzed_op in self.analyzed_ops:
            if analyzed_op.pdl_op == op:
                return analyzed_op
        return None

    def terminator_analysis(self):
        """
        
        """
        # We can have lost terminators when a terminator is erased
        # of replaced by a non-terminator
        # There is no way to generate ops after an existing terminator,
        # so analyzing erasures and replacements is sufficient.

        for terminator in self.terminator_matches:
            if terminator.erased_by and terminator.replaced_by is None:
                # Handle special case where terminator is erased but it is
                # the matched operation and before erasure a new terminator
                # is created. (creating new terminator after the erasure
                # segfaults in MLIR)
                if terminator.pdl_op == self.root_op:
                    # get the pdl.OperationOp before the erasure and check
                    # whether it is a terminator
                    assert self.rewrite_op.body
                    erase_op_idx = self.rewrite_op.body.ops.index(
                        terminator.erased_by)
                    if erase_op_idx > 0 and isinstance(
                            possible_new_terminator :=
                        (self.rewrite_op.body.ops[erase_op_idx - 1]),
                            pdl.OperationOp):
                        if (
                                possible_new_terminator_analysis :=
                                self.get_analysis_for_pdl_op(
                                    possible_new_terminator)
                        ) and possible_new_terminator_analysis.is_terminator:
                            continue

                debug(f"Terminator was erased: {terminator}!")
                self._add_analysis_result_to_op(terminator.pdl_op,
                                                "terminator_erased")
            elif replacement := terminator.replaced_by:
                if isinstance(replacement, AnalyzedPDLOperation):
                    if not replacement.is_terminator:
                        self._add_analysis_result_to_op(
                            terminator.pdl_op,
                            "terminator_replaced_with_non_terminator")
                        debug(
                            f"Terminator might be replaced by non-terminator: {terminator}!"
                        )
                else:
                    raise Exception(
                        "We currently can't handle pdl.Replace with a list of SSAValues as replacement!"
                    )

        debug("Terminator Analysis finished.")

    def def_use_analysis(self):
        """
        
        """

        debug("Dominance Analysis not implemented!")
        # analyze dominance of generated ops
        # for op in self.generated_ops:
        #     for operand in op.operands:
        #         if operand.op in self.matched_ops:
        #             self._add_analysis_result_to_op(op.pdl_op,
        #                                             "dominance_broken")

        # analyze dominance changes through erasures

    def _trace_match_operation_op(
            self, pdl_operation_op: pdl.OperationOp) -> AnalyzedPDLOperation:
        """
        Gather information about a pdl.OperationOp and its operands in 
        the matching part of a pdl.PatternOp.
        """
        # get matched operation type6
        if (name := pdl_operation_op.opName) and self._get_op_named(name.data):
            op_type = self._get_op_named(name.data)
            assert op_type is not None
        else:
            op_type = Operation

        # trace operands
        analyzed_operands: list[AnalyzedPDLOperation] = []
        for operand in pdl_operation_op.operands:
            # operands of PDL ops are always OpResults
            if not isinstance(operand, OpResult):
                raise Exception(
                    "Operands of PDL matching ops are always OpResults! The IR is inconsistent here!"
                )
            if analyzed_operand := self._trace_matching_op(operand.op):
                analyzed_operands.append(analyzed_operand)

        analyzed_pdl_op = AnalyzedPDLOperation(pdl_operation_op, op_type)
        analyzed_pdl_op.use_info.dominated_by.extend(analyzed_operands)
        self.analyzed_ops.append(analyzed_pdl_op)
        return analyzed_pdl_op

    def _trace_matching_op(self,
                           pdl_op: Operation) -> AnalyzedPDLOperation | None:
        """
        Gather information about a pdl operation the matching part of a pdl.PatternOp.
        Returns the analyzed operation if `pdl_op` stems from pdl.OperationOp or pdl.ResultOp.
        """
        if pdl_op in self.visited_ops:
            debug(f"tracing lhs: op {pdl_op.name} Already visited!")
            return
        # TODO: we want to use a match statement here. Damn you YAPF!
        # trace differently depending on which PDL op we encounter here
        if isinstance(pdl_op, pdl.OperationOp):
            self.visited_ops.add(pdl_op)
            return self._trace_match_operation_op(pdl_op)
        elif isinstance(pdl_op, pdl.ResultOp):
            used_op_result: SSAValue = pdl_op.parent_
            if not isinstance(used_op_result, OpResult) or not isinstance(
                    used_op_result.op, pdl.OperationOp):
                raise Exception(
                    "pdl.ResultOp must have the result of pdl.OperationOp as operand!"
                )
            used_op: pdl.OperationOp = used_op_result.op
            self.visited_ops.add(pdl_op)
            return self._trace_matching_op(used_op)
        elif isinstance(pdl_op, pdl.AttributeOp):
            debug(f"lhs: Found attribute: {pdl_op.name}")
        elif isinstance(pdl_op, pdl.TypeOp):
            debug(f"lhs: Found type: {pdl_op.name}")
        elif isinstance(pdl_op, pdl.OperandOp):
            debug(f"lhs: Found operand: {pdl_op.name}")
        else:
            debug(f"lhs: unsupported PDL op: {pdl_op.name}")

        self.visited_ops.add(pdl_op)

    def _analyze_pdl_rewrite_op(self, pdl_rewrite_op: pdl.RewriteOp):
        if pdl_rewrite_op.body:
            for rhs_op in pdl_rewrite_op.body.ops:
                self._analyze_rhs_op(rhs_op)

    def _analyze_rhs_op(self,
                        rhs_op: Operation) -> AnalyzedPDLOperation | None:
        if rhs_op in self.visited_ops:
            debug(f"tracing rhs: op {rhs_op.name} Already visited!")
            return
        if isinstance(rhs_op, pdl.OperationOp):
            self.visited_ops.add(rhs_op)
            return self._trace_generate_new_op(rhs_op)
        elif isinstance(rhs_op, pdl.ResultOp):
            used_op_result: SSAValue = rhs_op.parent_
            if not isinstance(used_op_result, OpResult) or not isinstance(
                    used_op_result.op, pdl.OperationOp):
                raise Exception(
                    "pdl.ResultOp must have the result of pdl.OperationOp as operand!"
                )
            used_op: pdl.OperationOp = used_op_result.op
            self.visited_ops.add(rhs_op)
            return self._analyze_rhs_op(used_op)
        elif isinstance(rhs_op, pdl.TypeOp):
            debug(f"rhs: Found type: {rhs_op.name}")
        elif isinstance(rhs_op, pdl.AttributeOp):
            debug(f"rhs: Found attribute: {rhs_op.name}")
        elif isinstance(rhs_op, pdl.ReplaceOp):
            debug(f"rhs: Found replacement: {rhs_op.name}")
            # For now only handle the case where the replacement is a single op
            if len(rhs_op.operands) != 2 or not all([
                    isinstance(operand.typ, pdl.OperationType)
                    for operand in rhs_op.operands
            ]):
                raise Exception("Replacement must be a single op for now!")
            assert isinstance(rhs_op.opValue, OpResult)
            assert isinstance(rhs_op.opValue.op, pdl.OperationOp)
            if (analyzed_op := self.get_analysis_for_pdl_op(
                    rhs_op.opValue.op)) is None:
                raise Exception("Unknown pdl.Operation to be replaced!")
            if rhs_op.replOperation:
                assert isinstance(rhs_op.replOperation, OpResult)
                assert isinstance(rhs_op.replOperation.op, pdl.OperationOp)
                if (analyzed_repl_op := self.get_analysis_for_pdl_op(
                        rhs_op.replOperation.op)) is None:
                    raise Exception("Unknown pdl.Operation to be replaced!")
                analyzed_op.replaced_by = analyzed_repl_op
            elif rhs_op.replValues:
                analyzed_op.replaced_by = rhs_op.replValues
        elif isinstance(rhs_op, pdl.EraseOp):
            assert isinstance(rhs_op.opValue, OpResult)
            assert isinstance(rhs_op.opValue.op, pdl.OperationOp)
            if (analyzed_op := self.get_analysis_for_pdl_op(
                    rhs_op.opValue.op)) is None:
                raise Exception("Unknown pdl.Operation to be erased!")
            analyzed_op.erased_by = rhs_op
        else:
            raise Exception(f"Unsupported PDL op: {rhs_op.name}")
        self.visited_ops.add(rhs_op)

    def _trace_generate_new_op(
        self,
        new_op_op: pdl.OperationOp,
    ) -> AnalyzedPDLOperation:
        # get matched operation type
        if (name := new_op_op.opName) and self._get_op_named(name.data):
            op_type = self._get_op_named(name.data)
            assert op_type is not None
        else:
            op_type = Operation

        self.analyzed_ops.append(analyzed_op := AnalyzedPDLOperation(
            new_op_op, op_type=op_type, matched=False))

        # analyze operands
        analyzed_operands: list[AnalyzedPDLOperation] = []
        for operand in new_op_op.operands:
            # operands of PDL matching ops are always OpResults
            if not isinstance(operand, OpResult):
                raise Exception(
                    "Operands of PDL matching ops are always OpResults! The IR is inconsistent here!"
                )
            if (analyzed_operand := self._analyze_rhs_op(operand.op)):
                analyzed_operands.append(analyzed_operand)

        # TODO: do we want to just add this or better check this?
        analyzed_op.use_info.dominated_by.extend(analyzed_operands)
        return analyzed_op

    def _get_op_named(self, name: str) -> Type[Operation] | None:
        return self.context.get_optional_op(name)

    def _is_terminator(self, op_type: Type[Operation]):
        if issubclass(op_type, TerminatorOp):
            debug("Found terminator:")
        else:
            debug("Found non-terminator:")

    @staticmethod
    def _add_analysis_result_to_op(op: Operation, result: str):
        analysis_results: list[StringAttr] = []
        if "pdl_analysis" in op.attributes and isinstance(
            (tmp := op.attributes["pdl_analysis"]), ArrayAttr):
            # We can ignore the type issues here since we actually check all the types.
            # Only the type checker doesn't get that.
            if not all([isinstance(attr, StringAttr)
                        for attr in tmp]):  # type: ignore
                raise Exception(
                    "pdl_analysis attribute must be an array of strings!")
            analysis_results.extend(
                op.attributes["pdl_analysis"].data)  # type: ignore
        analysis_results.append(StringAttr(result))
        op.attributes["pdl_analysis"] = ArrayAttr(analysis_results)


def pdl_analysis_pass(ctx: MLContext, prog: ModuleOp):
    for op in prog.ops:
        if isinstance(op, pdl.PatternOp):
            debug(f"Found pattern:{op.name}")
            analysis = PDLAnalysis(ctx, op)
            analysis.terminator_analysis()


if __name__ == '__main__':
    from xdsl.parser import Parser, Source
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.pdl import PDL

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(MemRef)
    ctx.register_dialect(Affine)
    ctx.register_dialect(Scf)
    ctx.register_dialect(Cf)
    ctx.register_dialect(CMath)
    ctx.register_dialect(IRDL)
    ctx.register_dialect(LLVM)
    ctx.register_dialect(Vector)
    ctx.register_dialect(MPI)
    ctx.register_dialect(GPU)
    ctx.register_dialect(PDL)

    #     prog = """
    #     "builtin.module"() ({
    #   "pdl.pattern"() ({
    #     %0 = "pdl.attribute"() : () -> !pdl.attribute
    #     %1 = "pdl.type"() : () -> !pdl.type
    #     %2 = "pdl.operation"(%0, %1) {attributeValueNames = ["attr"], operand_segment_sizes = array<i32: 0, 1, 1>} : (!pdl.attribute, !pdl.type) -> !pdl.operation
    #     %3 = "pdl.result"(%2) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    #     %4 = "pdl.operand"() : () -> !pdl.value
    #     %5 = "pdl.operation"(%3, %4) {attributeValueNames = [], opName = "arith.constant", operand_segment_sizes = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
    #     "pdl.rewrite"(%5) ({
    #     }) {name = "rewriter", operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
    #   }) {benefit = 1 : i16, sym_name = "operations"} : () -> ()
    # }) : () -> ()
    #     """

    #     prog = """
    #     "builtin.module"() ({
    #   "pdl.pattern"() ({
    #     %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    #     %1 = "pdl.type"() : () -> !pdl.type
    #     %2 = "pdl.operation"(%0, %1) {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
    #     "pdl.rewrite"(%2) ({
    #       %3 = "pdl.type"() : () -> !pdl.type
    #       %4 = "pdl.operation"(%0, %3) {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
    #       "pdl.replace"(%2, %4) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    #     }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
    #   }) {benefit = 1 : i16, sym_name = "infer_type_from_operation_replace"} : () -> ()
    # }) : () -> ()
    #     """

    prog = """
    "builtin.module"() ({
    "pdl.pattern"() ({
        %0 = "pdl.operation"() {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
        // CHECK: "pdl.operation"()
        // CHECK-SAME: "terminator_replaced_with_non_terminator"
        "pdl.rewrite"(%0) ({
          "pdl.erase"(%0) : (!pdl.operation, !pdl.operation) -> ()
        }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
    }) {benefit = 1 : i16, sym_name = "invalid_terminator_replacement"} : () -> ()
    }) : () -> ()
    """

    # TODO: here 5 is dominated by 4. Why did the system add this?
    erasing_and_adding_terminator = """
    "builtin.module"() ({
        "pdl.pattern"() ({
      %0 = "pdl.type"() : () -> !pdl.type
      %1 = "pdl.attribute"() : () -> !pdl.attribute
      %2 = "pdl.operation"(%1, %0) {attributeValueNames = ["value"], opName = "custom.const", operand_segment_sizes = array<i32: 0, 1, 1>} : (!pdl.attribute, !pdl.type) -> !pdl.operation
      %3 = "pdl.result"(%2) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      %4 = "pdl.operation"(%3) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 0>} : (!pdl.value) -> !pdl.operation
      "pdl.rewrite"(%4) ({
        %5 = "pdl.operation"() {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
        "pdl.erase"(%4) : (!pdl.operation) -> ()
      }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
    }) {benefit = 3 : i16} : () -> ()
    }) : () -> ()
    """

    parser = Parser(ctx=ctx,
                    prog=erasing_and_adding_terminator,
                    source=Source.MLIR)
    program = parser.parse_op()
    assert isinstance(program, ModuleOp)
    pdl_analysis_pass(ctx, program)

    printer = Printer(target=Printer.Target.MLIR)
    printer.print_op(program)