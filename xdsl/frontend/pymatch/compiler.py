"""
Compile a restricted, typed Python matcher and rewrite to Match and PDLInterp.

Matchers support assignments, fixed IRDL operand/result accessors, required
attributes/properties, SSAValue.owner, failure guards, and a final capture tuple.
Guards are `if not isinstance(value, OpClass): return None` or
`if attribute != constant_attribute: return None`. Attribute constructors run at
compile time. Rewrites support replacement with existing SSA values, not creation
of new operations. Unsupported syntax is diagnosed rather than executed or omitted.

This prototype assumes verified payload IR with canonical operation classes and
no attribute/property name collisions: Match/PDLInterp share one lookup namespace.
It retains Python classes while compiling, then erases them to PDL handle types.
The existing helper-based PyAST frontend remains available separately.
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass
from inspect import getclosurevars, signature
from types import UnionType
from typing import NoReturn, Union, get_args, get_origin, get_type_hints

from xdsl.dialects import builtin, match, pdl, pdl_interp
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import Attribute, Block, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.irdl import IRDLOperation, OptAttributeDef, OptPropertyDef, VariadicDef
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class _Handle:
    value: SSAValue
    python_type: type


class _Compiler:
    def __init__(
        self, function: Callable[..., object], block: Block, *, matching: bool
    ):
        self.info = PyASTContext.get_func_info(function)
        ast.increment_lineno(self.info.ast, function.__code__.co_firstlineno - 1)
        closure = getclosurevars(function)
        self.globals: dict[str, object] = {
            **closure.builtins,
            **self.info.globals,
            **closure.nonlocals,
        }
        self.hints = get_type_hints(function)
        self.block = block
        self.matching = matching
        self.bindings: dict[str, object] = {}
        args = self.info.ast.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
            self.error(self.info.ast, "expected positional arguments without defaults")
        self.parameters = args.posonlyargs + args.args
        self.locals = {
            node.id
            for node in ast.walk(self.info.ast)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        } | {arg.arg for arg in self.parameters}

    def error(self, node: ast.AST, message: str) -> NoReturn:
        raise CodeGenerationException(
            self.info.file,
            getattr(node, "lineno", self.info.ast.lineno),
            getattr(node, "col_offset", 0),
            message,
        )

    def emit(self, op: Operation) -> SSAValue:
        self.block.add_op(op)
        return op.results[0]

    def unwrap(self, value: SSAValue) -> SSAValue:
        assert isa(value, SSAValue[match.OptionalType])
        return self.emit(match.IsNotNullOp(value))

    def check_operation_type(self, value: SSAValue, cls: object, node: ast.AST) -> type:
        if cls is Operation:
            return Operation
        if (
            not isinstance(cls, type)
            or not issubclass(cls, IRDLOperation)
            or "get_irdl_definition" not in cls.__dict__
        ):
            self.error(node, "expected Operation or a concrete IRDL operation class")
        self.block.add_op(match.HasNameOp(value, cls.name))
        return cls

    def handle_type(self, annotation: object) -> TypeAttribute:
        cls = get_origin(annotation) or annotation
        if isinstance(cls, type):
            for python_type, ir_type in (
                (Operation, pdl.OperationType()),
                (SSAValue, pdl.ValueType()),
                (Attribute, pdl.AttributeType()),
            ):
                if issubclass(cls, python_type):
                    return ir_type
        self.error(
            self.info.ast, "expected a fixed, flat tuple of payload handle types"
        )

    def field(self, base: _Handle, node: ast.Attribute) -> _Handle:
        if base.value.type == pdl.ValueType() and node.attr == "owner":
            if not self.matching:
                self.error(node, "SSAValue.owner is only supported in matchers")
            # A block argument's owner is a Block, not a failed match. Keep the
            # optional operation until an isinstance guard requires it to exist.
            value = self.emit(match.GetDefiningOpOp(base.value))
            return _Handle(value, Operation)
        if not issubclass(base.python_type, IRDLOperation) or isinstance(
            base.value.type, match.OptionalType
        ):
            self.error(node, "field access requires a concrete IRDL operation type")
        definition = base.python_type.get_irdl_definition()
        for fields, constructor, python_type in (
            (definition.operands, match.GetOperandOp, SSAValue),
            (definition.results, match.GetResultOp, OpResult),
        ):
            for index, (name, _) in enumerate(fields):
                if name != node.attr:
                    continue
                if any(
                    isinstance(field, VariadicDef) for _, field in fields[: index + 1]
                ):
                    self.error(
                        node,
                        "optional/variadic fields and dynamic offsets are unsupported",
                    )
                if self.matching:
                    value = self.unwrap(self.emit(constructor(index, base.value)))
                else:
                    getter = (
                        pdl_interp.GetOperandOp
                        if constructor is match.GetOperandOp
                        else pdl_interp.GetResultOp
                    )
                    value = self.emit(getter(index, base.value))
                return _Handle(value, python_type)
        if accessor := definition.accessor_names.get(node.attr):
            name, kind = accessor
            field = (
                definition.attributes if kind == "attribute" else definition.properties
            )[name]
            if isinstance(field, OptAttributeDef | OptPropertyDef):
                self.error(
                    node, "optional/defaulted attribute accessors are unsupported"
                )
            if self.matching:
                value = self.unwrap(self.emit(match.GetAttributeOp(name, base.value)))
            else:
                value = self.emit(pdl_interp.GetAttributeOp(name, base.value))
            return _Handle(value, Attribute)
        self.error(
            node, f"unsupported IRDL field: {base.python_type.__name__}.{node.attr}"
        )

    def expression(self, node: ast.expr, *, constant: bool = False) -> object:
        if isinstance(node, ast.Name):
            if node.id in self.locals:
                if node.id not in self.bindings:
                    self.error(node, f"unbound local: {node.id}")
                value = self.bindings[node.id]
            elif node.id in self.globals:
                value = self.globals[node.id]
            else:
                self.error(node, f"unknown name: {node.id}")
            if constant and isinstance(value, _Handle):
                self.error(node, "expected a compile-time constant")
            return value
        if isinstance(node, ast.Attribute):
            base = self.expression(node.value, constant=constant)
            if isinstance(base, _Handle):
                return self.field(base, node)
            return getattr(base, node.attr)
        if isinstance(node, ast.Tuple | ast.List):
            values = tuple(
                self.expression(element, constant=constant) for element in node.elts
            )
            return values if isinstance(node, ast.Tuple) else list(values)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            return ast.literal_eval(node)
        if isinstance(node, ast.Call):
            constructor = self.expression(node.func, constant=True)
            if not isinstance(constructor, type) or not issubclass(
                constructor, Attribute
            ):
                self.error(
                    node, "only Attribute constructors may be called in expressions"
                )
            if any(keyword.arg is None for keyword in node.keywords):
                self.error(node, "keyword unpacking is unsupported")
            args = [self.expression(arg, constant=True) for arg in node.args]
            kwargs = {
                keyword.arg: self.expression(keyword.value, constant=True)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            return constructor(*args, **kwargs)
        self.error(node, f"unsupported expression: {type(node).__name__}")

    def guard(self, node: ast.If) -> None:
        if (
            node.orelse
            or len(node.body) != 1
            or not isinstance(failure := node.body[0], ast.Return)
            or (
                failure.value is not None
                and not (
                    isinstance(failure.value, ast.Constant)
                    and failure.value.value is None
                )
            )
        ):
            self.error(node, "expected a failure guard with only 'return None'")
        test = node.test
        if (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(call := test.operand, ast.Call)
            and self.expression(call.func, constant=True) is isinstance
            and len(call.args) == 2
            and not call.keywords
            and isinstance(target := call.args[0], ast.Name)
        ):
            value = self.expression(target)
            if not isinstance(value, _Handle) or not issubclass(
                value.python_type, Operation
            ):
                self.error(
                    target, "isinstance guards require an operation or SSAValue.owner"
                )
            unwrapped = value.value
            if isinstance(unwrapped.type, match.OptionalType):
                unwrapped = self.unwrap(unwrapped)
            cls = self.check_operation_type(
                unwrapped, self.expression(call.args[1], constant=True), call
            )
            self.bindings[target.id] = _Handle(unwrapped, cls)
            return
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.NotEq)
        ):
            lhs = self.expression(test.left)
            rhs = self.expression(test.comparators[0])
            if isinstance(lhs, Attribute) and isinstance(rhs, _Handle):
                lhs, rhs = rhs, lhs
            if (
                isinstance(lhs, _Handle)
                and lhs.value.type == pdl.AttributeType()
                and isinstance(rhs, Attribute)
            ):
                self.block.add_op(match.HasAttrValueOp(lhs.value, rhs))
                return
        self.error(
            test,
            "expected an isinstance or constant-attribute inequality failure guard",
        )

    def replace(self, node: ast.expr) -> None:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and self.bindings.get(node.func.value.id) is PatternRewriter
            and node.func.attr == "replace"
        ):
            self.error(node, "expected rewriter.replace with existing SSA values")
        if any(keyword.arg is None for keyword in node.keywords):
            self.error(node, "keyword unpacking is unsupported")
        args = [self.expression(arg) for arg in node.args]
        kwargs = {
            kw.arg: self.expression(kw.value)
            for kw in node.keywords
            if kw.arg is not None
        }
        try:
            bound = signature(PatternRewriter.replace).bind(None, *args, **kwargs)
        except TypeError as exc:
            self.error(node, str(exc))
        bound.apply_defaults()
        target = bound.arguments["op"]
        values = bound.arguments["new_results"]
        if (
            not isinstance(target, _Handle)
            or target.value.type != pdl.OperationType()
            or bound.arguments["new_ops"] not in ((), [])
            or bound.arguments["safe_erase"] is not True
            or not isa(values, tuple[_Handle, ...] | list[_Handle])
            or any(value.value.type != pdl.ValueType() for value in values)
        ):
            self.error(
                node,
                "replacement requires an operation, no new ops, and existing SSA values",
            )
        self.block.add_op(
            pdl_interp.ReplaceOp(target.value, [value.value for value in values])
        )

    def body(self) -> tuple[_Handle, ...]:
        statements = self.info.ast.body
        for index, node in enumerate(statements):
            if isinstance(node, ast.Pass) or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(target := node.targets[0], ast.Name)
            ):
                self.bindings[target.id] = self.expression(node.value)
            elif isinstance(node, ast.If) and self.matching:
                self.guard(node)
            elif isinstance(node, ast.Expr) and not self.matching:
                self.replace(node.value)
            elif isinstance(node, ast.Return) and index == len(statements) - 1:
                value = None if node.value is None else self.expression(node.value)
                if self.matching and isa(value, tuple[_Handle, ...]):
                    return value
                if not self.matching and value is None:
                    return ()
                self.error(
                    node, "expected a final capture tuple (or None in the rewrite)"
                )
            else:
                self.error(node, f"unsupported statement: {type(node).__name__}")
        if self.matching:
            self.error(self.info.ast, "expected a final capture tuple")
        return ()


def compile_pattern(
    matcher: Callable[..., object], rewrite: Callable[..., None]
) -> builtin.ModuleOp:
    """Compile both functions, rejecting syntax outside the prototype's subset."""
    match_op = match.MatcherOp()
    match_compiler = _Compiler(matcher, match_op.body.block, matching=True)
    if len(match_compiler.parameters) != 1:
        match_compiler.error(match_compiler.info.ast, "expected one root argument")
    root_name = match_compiler.parameters[0].arg
    root = match_op.body.block.args[0]
    root.name_hint = root_name
    root_type = match_compiler.check_operation_type(
        root, match_compiler.hints.get(root_name), match_compiler.info.ast
    )
    match_compiler.bindings[root_name] = _Handle(root, root_type)
    captures_type = match_compiler.hints.get("return")
    if get_origin(captures_type) in (Union, UnionType):
        alternatives = get_args(captures_type)
        captures_type = next(
            (arg for arg in alternatives if arg is not type(None)), None
        )
        if len(alternatives) != 2 or type(None) not in alternatives:
            match_compiler.error(match_compiler.info.ast, "expected tuple[...] | None")
    if get_origin(captures_type) is not tuple:
        match_compiler.error(match_compiler.info.ast, "expected tuple[...] | None")
    types = get_args(captures_type)
    ir_types = tuple(match_compiler.handle_type(t) for t in types)
    captures = match_compiler.body()
    if tuple(c.value.type for c in captures) != ir_types:
        match_compiler.error(
            match_compiler.info.ast, "capture tuple does not match its annotation"
        )

    rewrite_op = pdl_interp.FuncOp(rewrite.__name__, (ir_types, []))
    rewrite_compiler = _Compiler(rewrite, rewrite_op.body.block, matching=False)
    if rewrite.__name__ == "matcher":
        rewrite_compiler.error(
            rewrite_compiler.info.ast,
            "'matcher' is reserved for the lowered entry point",
        )
    parameters = rewrite_compiler.parameters
    if (
        len(parameters) != len(types) + 1
        or rewrite_compiler.hints.get(parameters[-1].arg) is not PatternRewriter
    ):
        rewrite_compiler.error(
            rewrite_compiler.info.ast,
            "expected captures followed by a PatternRewriter argument",
        )
    if rewrite_compiler.hints.get("return") is not type(None):
        rewrite_compiler.error(
            rewrite_compiler.info.ast, "expected a None rewrite return annotation"
        )
    for parameter, value, capture_type in zip(
        parameters, rewrite_op.body.block.args, types
    ):
        expected = rewrite_compiler.hints.get(parameter.arg)
        capture_cls = get_origin(capture_type) or capture_type
        expected_cls = get_origin(expected) or expected
        if not isinstance(expected_cls, type) or not issubclass(
            capture_cls, expected_cls
        ):
            rewrite_compiler.error(
                parameter, "rewrite argument is incompatible with its capture type"
            )
        value.name_hint = parameter.arg
        rewrite_compiler.bindings[parameter.arg] = _Handle(value, expected_cls)
    rewrite_compiler.bindings[parameters[-1].arg] = PatternRewriter
    rewrite_compiler.body()
    rewrite_op.body.block.add_op(pdl_interp.FinalizeOp())
    match_op.body.block.add_op(
        match.SuccessOp(rewrite.__name__, 1, [c.value for c in captures])
    )
    module = builtin.ModuleOp([match_op, rewrite_op])
    module.verify()
    return module
