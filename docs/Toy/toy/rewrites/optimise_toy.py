from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, cast
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.interpreters.experimental.pdl import PDLMatcher

from xdsl.ir import (
    Attribute,
    AttributeInvT,
    Block,
    MLContext,
    OpResult,
    Operation,
    OperationInvT,
    SSAValue,
)
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ArrayAttr,
    Float64Type,
    FloatAttr,
    ModuleOp,
    TensorType,
)
from xdsl.irdl import AttrConstraint, BaseAttr, EqAttrConstraint
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    AnonymousImplicitBuilderRewritePattern,
    ImplicitBuilderRewritePattern,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    implicit_rewriter,
)
from xdsl.transforms.dead_code_elimination import dce
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

from ..dialects.toy import (
    ConstantOp,
    ReshapeOp,
    TensorTypeF64,
    TransposeOp,
    UnrankedTensorTypeF64,
)


@dataclass(frozen=True)
class MatcherAttribute(Attribute):
    name = "matcher.attribute"
    id: str
    constraint: AttrConstraint
    _value: Attribute | None = None

    def match(self, other: Attribute) -> bool:
        if self._value is None:
            try:
                self.constraint.verify(other, {})
                setattr(self, "_value", other)
                return True
            except VerifyException:
                return False
        else:
            return self._value == other

    def unmatch(self) -> bool:
        unmatched = self._value is not None
        setattr(self, "_value", None)
        return unmatched


class MatcherSSAValue(SSAValue):
    _value: SSAValue | None = None

    def match(self, other: SSAValue) -> bool:
        if self._value is None:
            self._value = other
            return True
        else:
            return self._value == other

    def unmatch(self) -> bool:
        unmatched = self._value is not None
        self._value = None
        return unmatched

    @property
    def owner(self) -> Operation | Block:
        assert False, "Attempting to get the owner of a `MatcherSSAValue`"

    def __eq__(self, other: object) -> bool:
        return self is other

    # This might be problematic, as the superclass is not hashable ...
    def __hash__(self) -> int:  # type: ignore
        return id(self)


class MatchBuilder(Builder):
    def insert(self, op: OperationInvT) -> OperationInvT:
        # Rewrite all match attributes and values to the real deal
        assert False
        return op


class MatcherRewrite(RewritePattern):
    root: Operation
    _match_and_build: Callable[[Operation, PatternRewriter], Sequence[SSAValue] | None]

    def __init__(
        self,
        root: Operation,
        match_and_build: Callable[
            [Operation, PatternRewriter], Sequence[SSAValue] | None
        ],
    ):
        super().__init__()
        self.root = root
        self._match_and_build = match_and_build

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if op.parent is None:
            return

        with ImplicitBuilder(MatchBuilder(op.parent, op)):
            new_results = self._match_and_build(op, rewriter)
            if new_results is not None:
                rewriter.replace_matched_op([], new_results)

    @staticmethod
    def build(func: Callable[[Matcher], MatcherRewrite]) -> MatcherRewrite:
        return func(Matcher())


class Matcher:
    var_id: int = 0
    match_attributes: list[MatcherAttribute] = []
    match_values: list[MatcherSSAValue] = []

    def value(self, typ: type[Attribute]) -> SSAValue:
        value = MatcherSSAValue(self.attribute(typ))
        self.match_values.append(value)
        return cast(SSAValue, value)

    def attribute(self, typ: type[AttributeInvT]) -> AttributeInvT:
        constraint = BaseAttr(typ)
        var_id = self.var_id
        self.var_id += 1
        attr = MatcherAttribute(f"{var_id}", constraint)
        self.match_attributes.append(attr)
        return cast(AttributeInvT, attr)

    def match_op(self, op: Operation, *, root: Operation) -> bool:
        assert False

    def unmatch(self):
        for attr in self.match_attributes:
            assert attr.unmatch()
        for value in self.match_values:
            assert value.unmatch()

    def rewrite(
        self,
        root: Operation,
    ) -> Callable[[Callable[[], Sequence[SSAValue] | None]], MatcherRewrite]:
        def impl(
            rewrite_func: Callable[[], Sequence[SSAValue] | None]
        ) -> MatcherRewrite:
            def match_and_build(
                op: Operation, rewriter: PatternRewriter
            ) -> Sequence[SSAValue] | None:
                res = None
                if self.match_op(op, root=root):
                    res = rewrite_func()
                    self.unmatch()
                return res

            return MatcherRewrite(root, match_and_build)

        return impl


@MatcherRewrite.build
def tt(matcher: Matcher) -> MatcherRewrite:
    arg = matcher.value(Attribute)
    t0 = matcher.attribute(TensorTypeF64 | UnrankedTensorTypeF64)
    t1 = matcher.attribute(TensorTypeF64 | UnrankedTensorTypeF64)
    transpose_0 = TransposeOp(arg, t0)
    transpose_1 = TransposeOp(transpose_0.res, t1)

    @matcher.rewrite(transpose_1)
    def rewrite() -> Sequence[SSAValue] | None:
        return (arg,)

    return rewrite


# @MatcherRewrite.build
def rr(matcher: Matcher) -> MatcherRewrite:
    arg = matcher.value(Attribute)
    typ_0 = matcher.attribute(TensorType[Float64Type])
    r_0 = ReshapeOp(arg, typ_0)
    typ_1 = matcher.attribute(TensorType[Float64Type])
    r_1 = ReshapeOp(r_0.res, typ_1)

    @matcher.rewrite(r_1)
    def rewrite() -> Sequence[SSAValue] | None:
        return ReshapeOp(arg, typ_1).results

    return rewrite


@implicit_rewriter
def simplify_redundant_transpose(op: TransposeOp):
    """Fold transpose(transpose(x)) -> x"""
    if isinstance(input := op.arg, OpResult) and isinstance(input.op, TransposeOp):
        return input.op.operands


@implicit_rewriter
def reshape_reshape(op: ReshapeOp):
    """Reshape(Reshape(x)) = Reshape(x)"""
    if isinstance(input := op.arg, OpResult) and isinstance(input.op, ReshapeOp):
        t = cast(TensorType[Float64Type], op.res.typ)
        new_op = ReshapeOp(input.op.arg, t)
        return new_op.results


@implicit_rewriter
def constant_reshape(op: ReshapeOp):
    """
    Reshaping a constant can be done at compile time
    """
    if isinstance(input := op.arg, OpResult) and isinstance(input.op, ConstantOp):
        assert isa(op.res.typ, TensorTypeF64)
        assert isa(input.op.value.data, ArrayAttr[FloatAttr[Float64Type]])

        new_value = DenseIntOrFPElementsAttr.create_dense_float(
            type=op.res.typ, data=input.op.value.data.data
        )
        new_op = ConstantOp(new_value)
        return new_op.results


class OptimiseToy(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(tt).rewrite_module(op)
        PatternRewriteWalker(reshape_reshape).rewrite_module(op)
        PatternRewriteWalker(constant_reshape).rewrite_module(op)
        dce(op)
