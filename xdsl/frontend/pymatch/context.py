"""Compile straight-line Python matchers through PyAST's registered IR builders."""

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import signature
from typing import get_args, get_origin, get_type_hints

from xdsl.context import Context
from xdsl.dialects import builtin, func, match, pdl, pdl_interp
from xdsl.frontend import pymatch
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.program import P, PyASTProgram, R
from xdsl.frontend.pyast.utils.exceptions import (
    CodeGenerationException,
    FrontendProgramException,
)
from xdsl.ir import Attribute, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.transforms.desymref import FrontendDesymrefyPass
from xdsl.utils.hints import isa


@irdl_op_definition
class _CallOp(IRDLOperation):
    """Keep helper arguments as SSA values until local bindings are resolved."""

    name = "pymatch.call"
    callee = prop_def(builtin.StringAttr)
    arguments = var_operand_def(pdl.AnyPDLTypeConstr)
    outputs = var_result_def(pdl.AnyPDLTypeConstr)

    def __init__(
        self,
        callee: str,
        arguments: Sequence[SSAValue],
        results: Sequence[Attribute] = (),
    ):
        super().__init__(
            operands=[arguments],
            result_types=[results],
            properties={"callee": builtin.StringAttr(callee)},
        )


@irdl_op_definition
class _TupleOp(IRDLOperation):
    """A flat, typed capture tuple; erased when lowering the final return."""

    name = "pymatch.tuple"
    elements = var_operand_def(pdl.AnyPDLTypeConstr)
    result = result_def(builtin.TupleType)

    def __init__(self, elements: Sequence[SSAValue]):
        types = tuple(element.type for element in elements)
        assert isa(types, tuple[TypeAttribute, ...])
        super().__init__(operands=[elements], result_types=[builtin.TupleType(types)])


def _constant(value: SSAValue) -> Attribute:
    if not isinstance(value, OpResult) or not isinstance(
        value.owner, pdl_interp.CreateAttributeOp
    ):
        raise FrontendProgramException(
            "expected a compile-time constant helper argument"
        )
    return value.owner.value


def _integer(value: SSAValue) -> int:
    attribute = _constant(value)
    if not isa(attribute, builtin.IntAttr):
        raise FrontendProgramException("expected a constant Python integer")
    return attribute.data


def _string(value: SSAValue) -> str:
    attribute = _constant(value)
    if not isa(attribute, builtin.StringAttr):
        raise FrontendProgramException("expected a constant Python string")
    return attribute.data


def _lower_call(op: _CallOp) -> None:
    args = op.arguments
    name = op.callee.data
    navigation: (
        match.GetOperandOp
        | match.GetResultOp
        | match.GetDefiningOpOp
        | match.GetAttributeOp
    )
    if name in ("get_operand", "get_result"):
        index = _integer(args[1])
        if index < 0:
            raise FrontendProgramException("expected a nonnegative index")
        constructor = match.GetOperandOp if name == "get_operand" else match.GetResultOp
        navigation = constructor(index, args[0])
    elif name == "get_defining_op":
        navigation = match.GetDefiningOpOp(args[0])
    elif name == "get_attribute":
        navigation = match.GetAttributeOp(_string(args[1]), args[0])
    else:
        if name == "has_name":
            lowered = match.HasNameOp(args[0], _string(args[1]))
        elif name in ("check_operand_count", "check_result_count"):
            count = _integer(args[1])
            if count < 0:
                raise FrontendProgramException("expected a nonnegative count")
            constructor = (
                match.CheckOperandCountOp
                if name == "check_operand_count"
                else match.CheckResultCountOp
            )
            lowered = constructor(args[0], count)
        elif name == "has_attr_value":
            expected = _constant(args[1])
            if not isa(expected, builtin.IntegerAttr):
                raise FrontendProgramException(
                    "expected an attribute constructed with IntegerAttr"
                )
            lowered = match.HasAttrValueOp(args[0], expected)
        elif name == "IntegerAttr":
            lowered = pdl_interp.CreateAttributeOp(
                builtin.IntegerAttr(_integer(args[0]), _integer(args[1]))
            )
        else:
            raise FrontendProgramException(f"unsupported match helper: {name}")
        Rewriter.replace_op(op, lowered)
        return
    # Python navigation fails immediately rather than returning an optional handle.
    Rewriter.replace_op(op, [navigation, match.IsNotNullOp(navigation.result)])


@dataclass(frozen=True)
class _LowerMatchPass(ModulePass):
    """Internal, Python-configured pass; the rewriter is cloned into each module."""

    name = "lower-pymatch"
    rewriter: pdl_interp.FuncOp

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        (function,) = op.ops
        assert isinstance(function, func.FuncOp)
        block = function.body.block
        for helper in tuple(block.ops):
            if isinstance(helper, _CallOp):
                _lower_call(helper)
        terminator = block.last_op
        assert isinstance(terminator, func.ReturnOp)
        if len(terminator.arguments) != 1 or not isinstance(
            captures := terminator.arguments[0].owner, _TupleOp
        ):
            raise FrontendProgramException("expected a flat capture tuple return")
        if captures.elements.types != self.rewriter.function_type.inputs.data:
            raise FrontendProgramException(
                "capture types do not match rewriter arguments"
            )
        Rewriter.replace_op(
            terminator,
            match.SuccessOp(self.rewriter.sym_name.data, 1, captures.elements),
        )
        for helper in reversed(tuple(block.ops)):
            if isinstance(helper, _TupleOp | pdl_interp.CreateAttributeOp):
                if any(result.uses for result in helper.results):
                    raise FrontendProgramException(
                        "only payload handles can be captured; constants must be used by checks"
                    )
                Rewriter.erase_op(helper)
        Rewriter.replace_op(function, match.MatcherOp(function.detach_region(0)))
        op.body.block.add_op(self.rewriter.clone())
        op.verify()


class PyMatchContext(PyASTContext):
    """
    Decorate a matcher taking one Operation and returning a fixed, flat typed tuple.

    Tuple elements may be Operation, SSAValue, or Attribute. Calls run as ordinary
    Python; `.module` compiles the matcher and includes a clone of the supplied
    PDLInterp rewriter. Rewriter arguments must match the tuple element types.

    Supported source forms are single-name assignments, registered helper calls,
    integer/string literals, and one final tuple return. IntegerAttr(value, width)
    constructs a constant attribute for a check. Literal helper arguments may be
    locally bound. Control flow, optional handles, payload mutation, and compiling
    Python rewrite functions are outside this prototype.
    """

    def __init__(self, rewriter: pdl_interp.FuncOp):
        if rewriter.sym_name.data == "matcher" or rewriter.function_type.outputs.data:
            raise FrontendProgramException(
                "expected a result-free rewriter not named 'matcher'"
            )
        super().__init__(
            post_transforms=[FrontendDesymrefyPass(), _LowerMatchPass(rewriter.clone())]
        )
        self.register_type(Operation, pdl.OperationType())
        self.register_type(SSAValue, pdl.ValueType())
        self.register_type(Attribute, pdl.AttributeType())
        self.register_tuple(_TupleOp)
        self.register_literal(
            int, lambda value: pdl_interp.CreateAttributeOp(builtin.IntAttr(value))
        )
        self.register_literal(
            str, lambda value: pdl_interp.CreateAttributeOp(builtin.StringAttr(value))
        )
        for helper in (
            pymatch.has_name,
            pymatch.check_operand_count,
            pymatch.check_result_count,
            pymatch.has_attr_value,
        ):
            self._register_helper(helper)
        self._register_helper(pymatch.get_operand, [pdl.ValueType()])
        self._register_helper(pymatch.get_result, [pdl.ValueType()])
        self._register_helper(pymatch.get_defining_op, [pdl.OperationType()])
        self._register_helper(pymatch.get_attribute, [pdl.AttributeType()])

        def integer_attr(value: SSAValue, value_type: SSAValue) -> _CallOp:
            return _CallOp("IntegerAttr", [value, value_type], [pdl.AttributeType()])

        self.register_function(builtin.IntegerAttr, integer_attr)

    def _register_helper(
        self, helper: Callable[..., object], results: Sequence[Attribute] = ()
    ) -> None:
        parameters = signature(helper)

        def build(*args: SSAValue, **kwargs: SSAValue) -> _CallOp:
            bound = parameters.bind(*args, **kwargs)
            return _CallOp(helper.__name__, tuple(bound.arguments.values()), results)

        self.register_function(helper, build)

    def parse_program(self, func: Callable[P, R]) -> PyASTProgram[P, R]:
        info = self._get_func_info(func)
        args = info.ast.args
        if (
            len(args.args) != 1
            or args.posonlyargs
            or args.kwonlyargs
            or args.vararg
            or args.kwarg
            or args.defaults
        ):
            raise FrontendProgramException("expected one positional root argument")
        for statement in info.ast.body:
            if not isinstance(statement, ast.Assign | ast.Expr | ast.Return | ast.Pass):
                raise CodeGenerationException(
                    info.file,
                    statement.lineno,
                    statement.col_offset,
                    "expected a straight-line matcher with assignments, calls, and a return",
                )
        hints = get_type_hints(func)
        types = {
            Operation: pdl.OperationType(),
            SSAValue: pdl.ValueType(),
            Attribute: pdl.AttributeType(),
        }
        captures = hints.get("return")
        if (
            captures is None
            or hints.get(args.args[0].arg) is not Operation
            or get_origin(captures) is not tuple
        ):
            raise FrontendProgramException(
                "expected Operation -> tuple of payload handles"
            )
        elements = get_args(captures)
        if any(element not in types for element in elements):
            raise FrontendProgramException(
                "expected a fixed, flat tuple of payload handle types"
            )
        assert info.ast.returns is not None
        if (
            self.type_registry.resolve_attribute(
                ast.unparse(info.ast.returns), info.globals
            )
            is None
        ):
            self.register_type(
                captures, builtin.TupleType(tuple(types[t] for t in elements))
            )
        return super().parse_program(func)
