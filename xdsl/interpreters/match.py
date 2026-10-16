"""Direct interpretation of straight-line, single-success Match patterns."""

from dataclasses import dataclass

from xdsl.dialects import match, pdl
from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.interpreter import PythonValues
from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.utils.exceptions import InterpretationError


@dataclass(frozen=True)
class MatchResult:
    """A successful match, without having executed its rewrite."""

    rewriter: SymbolRefAttr
    inputs: PythonValues


def run_matcher(matcher: match.MatcherOp, root: Operation) -> MatchResult | None:
    """
    Match without modifying payload IR. Return captures, or None on mismatch.

    Each invocation owns its SSA bindings. Failed predicates stop evaluation
    before assigning results; unsupported operations and programming errors raise.
    Only one final success is supported; benefits do not affect execution.
    """
    ops = tuple(matcher.body.block.ops)
    if (
        not ops
        or not isinstance(success := ops[-1], match.SuccessOp)
        or any(isinstance(op, match.SuccessOp) for op in ops[:-1])
    ):
        raise InterpretationError(
            "expected one match.success at the end of the matcher"
        )

    values: dict[SSAValue, Operation | SSAValue | Attribute | None] = {
        matcher.body.block.args[0]: root
    }
    for op in ops[:-1]:
        results = run_match_op(op, tuple(values[value] for value in op.operands))
        if results is None:
            return None
        values.update(zip(op.results, results, strict=True))
    return MatchResult(
        success.rewriter, tuple(values[value] for value in success.inputs)
    )


def run_match_op(op: Operation, args: PythonValues) -> PythonValues | None:
    """
    Evaluate a scalar Match operation on its operand values.

    Return SSA results, or None for a failed predicate. In particular, () is a
    successful check and (None,) is a successful navigation yielding a null handle.
    match.success is handled by run_matcher, not by this operation evaluator.
    """
    if isinstance(op, match.GetOperandOp | match.GetResultOp):
        (root,) = args
        assert isinstance(root, Operation)
        if op.index.value.data < 0 or op.result.type != match.OptionalType(
            pdl.ValueType()
        ):
            raise InterpretationError(
                f"{op.name} requires a nonnegative index and an optional value result"
            )
        index = op.index.value.data
        values = root.operands if isinstance(op, match.GetOperandOp) else root.results
        return (values[index] if index < len(values) else None,)

    if isinstance(op, match.GetAttributeOp):
        (root,) = args
        assert isinstance(root, Operation)
        if op.result.type != match.OptionalType(pdl.AttributeType()):
            raise InterpretationError(
                "get_attribute requires an optional attribute result"
            )
        # Match PDLInterp's lookup of both discardable attributes and properties.
        name = op.attribute_name.data
        return (root.attributes.get(name, root.properties.get(name)),)

    if isinstance(op, match.GetDefiningOpOp):
        if op.value.type != pdl.ValueType() or op.result.type != match.OptionalType(
            pdl.OperationType()
        ):
            raise InterpretationError(
                "get_defining_op requires a scalar value and an optional operation result"
            )
        (value,) = args
        assert isinstance(value, SSAValue)
        return (value.owner if isinstance(value, OpResult) else None,)

    if isinstance(op, match.IsNotNullOp):
        return None if args[0] is None else args

    if isinstance(op, match.HasNameOp):
        (root,) = args
        assert isinstance(root, Operation)
        return () if root.name == op.operation_name.data else None

    if isinstance(op, match.CheckOperandCountOp | match.CheckResultCountOp):
        (root,) = args
        assert isinstance(root, Operation)
        count = (
            len(root.operands)
            if isinstance(op, match.CheckOperandCountOp)
            else len(root.results)
        )
        expected = op.count.value.data
        matches = count >= expected if op.at_least is not None else count == expected
        return () if matches else None

    if isinstance(op, match.HasAttrValueOp):
        return () if args[0] == op.value else None

    raise InterpretationError(f"unsupported Match operation: {op.name}")
