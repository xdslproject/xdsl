"""Lower one straight-line Match pattern to the PDLInterp matcher entry point."""

from xdsl.context import Context
from xdsl.dialects import match, pdl, pdl_interp
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.traits import SymbolTable
from xdsl.utils.exceptions import PassFailedException


def lower_matcher(matcher: match.MatcherOp) -> pdl_interp.FuncOp:
    """
    Build a detached matcher function, leaving the source unchanged on failure.

    This prototype supports scalar navigation with natural result types and
    nonnegative indices, and exactly one success at the end of the body.
    These restrictions belong to the lowering, not the Match dialect verifier.
    """
    ops = tuple(matcher.body.block.ops)
    if (
        not ops
        or not isinstance(ops[-1], match.SuccessOp)
        or any(isinstance(op, match.SuccessOp) for op in ops[:-1])
    ):
        raise PassFailedException(
            "expected one match.success at the end of the matcher"
        )

    function = pdl_interp.FuncOp("matcher", ([pdl.OperationType()], []))
    block = function.body.block
    root = block.args[0]
    root.name_hint = matcher.body.block.args[0].name_hint
    values: dict[SSAValue, SSAValue] = {matcher.body.block.args[0]: root}
    failure = Block([pdl_interp.FinalizeOp()])
    matched_ops: list[SSAValue] = [root]
    root_kind: StringAttr | None = None

    for op in ops:
        lowered: Operation | None
        if isinstance(op, match.GetOperandOp | match.GetResultOp):
            if op.index.value.data < 0:
                raise PassFailedException(
                    f"{op.name}: negative indices are not supported"
                )
            constructor = (
                pdl_interp.GetOperandOp
                if isinstance(op, match.GetOperandOp)
                else pdl_interp.GetResultOp
            )
            lowered = constructor(op.index, values[op.input_op])
        elif isinstance(op, match.GetAttributeOp):
            lowered = pdl_interp.GetAttributeOp(op.attribute_name, values[op.input_op])
        elif isinstance(op, match.GetDefiningOpOp):
            if op.value.type != pdl.ValueType():
                raise PassFailedException(
                    f"{op.name}: only scalar values are supported"
                )
            lowered = pdl_interp.GetDefiningOpOp(values[op.value])
            matched_ops.append(lowered.results[0])
        else:
            lowered = None

        if lowered is not None:
            if op.results[0].type != match.OptionalType.new([lowered.results[0].type]):
                raise PassFailedException(
                    f"{op.name}: unsupported optional result type"
                )
            block.add_op(lowered)
            values[op.results[0]] = lowered.results[0]
            lowered.results[0].name_hint = op.results[0].name_hint
            lowered.location = op.location
            continue

        if isinstance(op, match.SuccessOp):
            block.add_op(
                pdl_interp.RecordMatchOp(
                    op.rewriter,
                    root_kind,
                    None,
                    op.benefit.value.data,
                    [values[value] for value in op.inputs],
                    matched_ops,
                    failure,
                )
            )
            continue

        continuation = Block()
        if isinstance(op, match.HasNameOp):
            lowered = pdl_interp.CheckOperationNameOp(
                op.operation_name, values[op.input_op], continuation, failure
            )
            if values[op.input_op] is root:
                root_kind = op.operation_name
        elif isinstance(op, match.CheckOperandCountOp | match.CheckResultCountOp):
            constructor = (
                pdl_interp.CheckOperandCountOp
                if isinstance(op, match.CheckOperandCountOp)
                else pdl_interp.CheckResultCountOp
            )
            lowered = constructor(
                values[op.input_op],
                op.count.value.data,
                continuation,
                failure,
                compareAtLeast=op.at_least is not None,
            )
        elif isinstance(op, match.HasAttrValueOp):
            lowered = pdl_interp.CheckAttributeOp(
                op.value, values[op.attribute], continuation, failure
            )
        elif isinstance(op, match.IsNotNullOp):
            value = values[op.optional_value]
            lowered = pdl_interp.IsNotNullOp(value, continuation, failure)
            values[op.unwrapped] = value
        else:
            raise PassFailedException(
                f"unsupported operation in scalar matcher: {op.name}"
            )

        lowered.location = op.location
        block.add_op(lowered)
        function.body.add_block(continuation)
        block = continuation

    function.body.add_block(failure)
    function.location = matcher.location
    return function


class ConvertMatchToPDLInterpPass(ModulePass):
    """
    Lower a single top-level scalar matcher to `pdl_interp.func @matcher`.

    Rewriter symbols are preserved. Multiple matchers and competing successes
    are deliberately unsupported until candidate selection is implemented.
    """

    name = "convert-match-to-pdl-interp"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        matchers = [child for child in op.walk() if isinstance(child, match.MatcherOp)]
        if len(matchers) != 1 or matchers[0].parent_op() is not op:
            raise PassFailedException("expected exactly one top-level match.matcher")
        matcher = matchers[0]
        if (
            existing := SymbolTable.lookup_symbol(op, "matcher")
        ) is not None and existing is not matcher:
            raise PassFailedException(
                "cannot create PDLInterp entry point: @matcher already exists"
            )
        Rewriter.replace_op(matcher, lower_matcher(matcher))
