from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple, Type
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

enable_prints = False


def debug(msg: str):
    if enable_prints:
        print(msg)


@dataclass
class AnalyzedPDLOperation:
    pdl_op: pdl.OperationOp
    op_type: Type[Operation]
    matched: bool = True
    erased: bool = False
    replaced_by: list[SSAValue] | AnalyzedPDLOperation | None = None

    @property
    def is_terminator(self):
        return issubclass(self.op_type, TerminatorOp)


class OpReplacement(NamedTuple):
    op: AnalyzedPDLOperation
    replacement: list[SSAValue] | AnalyzedPDLOperation


@dataclass
class PDLAnalysisContext:
    context: MLContext
    pattern_op: pdl.PatternOp
    root_op: Operation
    rewrite_op: pdl.RewriteOp
    analyzed_ops: list[AnalyzedPDLOperation] = field(default_factory=list)
    visited_ops: set[Operation] = field(default_factory=set)

    def __post_init__(self):
        self.visited_ops.add(self.rewrite_op)

    @property
    def matched_ops(self):
        return [op for op in self.analyzed_ops if op.matched]

    @property
    def erased_ops(self):
        return [op for op in self.analyzed_ops if op.erased]

    @property
    def terminator_matches(self):
        return [
            op for op in self.analyzed_ops
            if (op.is_terminator or op.op_type == Operation) and op.matched
        ]

    @property
    def generated_ops(self):
        return [op for op in self.analyzed_ops if not op.matched]

    def analysis_for_pdl_op(
            self, op: pdl.OperationOp) -> AnalyzedPDLOperation | None:
        for analyzed_op in self.analyzed_ops:
            if analyzed_op.pdl_op == op:
                return analyzed_op
        return None


def pdl_analysis_pass(ctx: MLContext, prog: ModuleOp):
    for op in prog.ops:
        if isinstance(op, pdl.PatternOp):
            analyze_pdl_pattern(op, ctx)


def analyze_pdl_pattern(op: pdl.PatternOp, ctx: MLContext):
    debug(f"Found pattern:{op.name}")
    analysis_context = populate_analysis_context(op, ctx)

    if len(op.body.ops) != len(analysis_context.visited_ops):
        if enable_prints:
            debug(
                f"{abs(len(op.body.ops) - len(analysis_context.visited_ops))} Unreachable PDL ops in pattern!"
            )
            debug("Full pattern:")
            printer = Printer()
            printer.print_op(op)
            debug("Unreachable ops:")
            for pdl_op in op.body.ops:
                if pdl_op not in analysis_context.visited_ops:
                    printer.print_op(pdl_op)

    analyze_pdl_rewrite_op(analysis_context.rewrite_op, analysis_context)

    terminator_analysis(analysis_context)


def terminator_analysis(analysis_context: PDLAnalysisContext):
    # We can have lost terminators when a terminator is erased
    # of replaced by a non-terminator
    # There is no way to generate ops after an existing terminator,
    # so analyzing erasures and replacements is sufficient.

    for terminator in analysis_context.terminator_matches:
        if terminator.erased and terminator.replaced_by is None:
            debug(f"Terminator was erased: {terminator}!")
            add_analysis_result_to_op(terminator.pdl_op, "terminator_erased")
        elif replacement := terminator.replaced_by:
            if isinstance(replacement, AnalyzedPDLOperation):
                if not replacement.is_terminator:
                    add_analysis_result_to_op(
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


def populate_analysis_context(op: pdl.PatternOp,
                              ctx: MLContext) -> PDLAnalysisContext:
    if isinstance(op.body.ops[-1], pdl.RewriteOp):
        rewrite_op: pdl.RewriteOp = op.body.ops[-1]
    else:
        raise Exception(
            "pdl.PatternOp must have a pdl.RewriteOp as terminator!")

    if isinstance(op.body.ops[-2], pdl.OperationOp):
        root_op: pdl.OperationOp = op.body.ops[-2]
    else:
        raise Exception("pdl.PatternOp must have a pdl.OperationOp as root!")

    analysis_context = PDLAnalysisContext(ctx, op, root_op, rewrite_op)
    trace_matching_op(root_op, analysis_context)

    return analysis_context


def analyze_pdl_rewrite_op(pdl_rewrite_op: pdl.RewriteOp,
                           analysis_ctx: PDLAnalysisContext):
    if pdl_rewrite_op.body:
        for rhs_op in pdl_rewrite_op.body.ops:
            analyze_rhs_op(rhs_op, analysis_ctx)


def get_op_named(name: str, context: MLContext) -> Type[Operation] | None:
    return context.get_optional_op(name)


def is_terminator(op_type: Type[Operation]):
    if issubclass(op_type, TerminatorOp):
        debug("Found terminator:")
    else:
        debug("Found non-terminator:")


def _trace_match_operation_op(pdl_operation_op: pdl.OperationOp,
                              analysis_ctx: PDLAnalysisContext):
    """
    Gather information about a pdl.OperationOp and its operands in 
    the matching part of a pdl.PatternOp.
    """
    # get matched operation type
    if (name := pdl_operation_op.opName) and get_op_named(name.data, ctx):
        op_type = get_op_named(name.data, ctx)
        assert op_type is not None
    else:
        op_type = Operation

    # trace operands
    for operand in pdl_operation_op.operands:
        # operands of PDL ops are always OpResults
        if not isinstance(operand, OpResult):
            raise Exception(
                "Operands of PDL matching ops are always OpResults! The IR is inconsistent here!"
            )
        trace_matching_op(operand.op, analysis_ctx)

    analyzed_pdl_op = AnalyzedPDLOperation(pdl_operation_op, op_type)
    analysis_ctx.analyzed_ops.append(analyzed_pdl_op)


def trace_matching_op(pdl_op: Operation, analysis_ctx: PDLAnalysisContext):
    """
    Gather information about a pdl operation the matching part of a pdl.PatternOp.
    """
    if pdl_op in analysis_ctx.visited_ops:
        debug(f"tracing lhs: op {pdl_op.name} Already visited!")
        return
    # TODO: we want to use a match statement here. Damn you YAPF!
    # trace differently depending on which PDL op we encounter here
    if isinstance(pdl_op, pdl.OperationOp):
        _trace_match_operation_op(pdl_op, analysis_ctx)
    elif isinstance(pdl_op, pdl.ResultOp):
        used_op_result: SSAValue = pdl_op.parent_
        if not isinstance(used_op_result, OpResult) or not isinstance(
                used_op_result.op, pdl.OperationOp):
            raise Exception(
                "pdl.ResultOp must have the result of pdl.OperationOp as operand!"
            )
        used_op: pdl.OperationOp = used_op_result.op
        _trace_match_operation_op(used_op, analysis_ctx)
    elif isinstance(pdl_op, pdl.AttributeOp):
        debug(f"lhs: Found attribute: {pdl_op.name}")
    elif isinstance(pdl_op, pdl.TypeOp):
        debug(f"lhs: Found type: {pdl_op.name}")
    elif isinstance(pdl_op, pdl.OperandOp):
        debug(f"lhs: Found operand: {pdl_op.name}")
    else:
        debug(f"lhs: unsupported PDL op: {pdl_op.name}")

    analysis_ctx.visited_ops.add(pdl_op)


def analyze_rhs_op(rhs_op: Operation, analysis_ctx: PDLAnalysisContext):
    if rhs_op in analysis_ctx.visited_ops:
        debug(f"tracing rhs: op {rhs_op.name} Already visited!")
        return
    if isinstance(rhs_op, pdl.OperationOp):
        trace_generate_new_op(rhs_op, analysis_ctx)
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
        if (analyzed_op := analysis_ctx.analysis_for_pdl_op(
                rhs_op.opValue.op)) is None:
            raise Exception("Unknown pdl.Operation to be replaced!")
        if rhs_op.replOperation:
            assert isinstance(rhs_op.replOperation, OpResult)
            assert isinstance(rhs_op.replOperation.op, pdl.OperationOp)
            if (analyzed_repl_op := analysis_ctx.analysis_for_pdl_op(
                    rhs_op.replOperation.op)) is None:
                raise Exception("Unknown pdl.Operation to be replaced!")
            analyzed_op.replaced_by = analyzed_repl_op
        elif rhs_op.replValues:
            analyzed_op.replaced_by = rhs_op.replValues
    else:
        debug(f"rhs: unsupported PDL op: {rhs_op.name}")
    analysis_ctx.visited_ops.add(rhs_op)


def trace_generate_new_op(new_op_op: pdl.OperationOp,
                          analysis_ctx: PDLAnalysisContext):
    # get matched operation type
    if (name := new_op_op.opName) and get_op_named(name.data,
                                                   analysis_ctx.context):
        op_type = get_op_named(name.data, analysis_ctx.context)
        assert op_type is not None
    else:
        op_type = Operation

    analysis_ctx.analyzed_ops.append(
        AnalyzedPDLOperation(new_op_op, op_type=op_type, matched=False))

    for operand in new_op_op.operands:
        # operands of PDL matching ops are always OpResults
        if not isinstance(operand, OpResult):
            raise Exception(
                "Operands of PDL matching ops are always OpResults! The IR is inconsistent here!"
            )
        analyze_rhs_op(operand.op, analysis_ctx)


def add_analysis_result_to_op(op: Operation, result: str):
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

    prog = """
    "builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.operation"(%0, %1) {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%2) ({
      %3 = "pdl.type"() : () -> !pdl.type
      %4 = "pdl.operation"(%0, %3) {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
      "pdl.replace"(%2, %4) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_operation_replace"} : () -> ()
}) : () -> ()
    """

    parser = Parser(ctx=ctx, prog=prog, source=Source.MLIR)
    program = parser.parse_op()
    assert isinstance(program, ModuleOp)
    pdl_analysis_pass(ctx, program)

    printer = Printer(target=Printer.Target.MLIR)
    printer.print_op(program)