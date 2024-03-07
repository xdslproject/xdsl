from abc import ABC
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import ClassVar, Callable, TypeVar

from dialects import builtin
from ir import MLContext

from xdsl.ir import Operation
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, PatternRewriteWalker
from xdsl.passes import ModulePass

class ConversionMode(IntEnum):
    Partial = auto()
    """
    Operations that are allowed after pass completion are legal ones and unknown ones.
    """
    Full = auto()
    """
    Only operations declared legal are allowed to exist after conversion.
    """

class ConversionPassError(ValueError):
    pass

_IS_CONVERSION_PATTERN = object()
_CONVERSION_INPUT_TYPE = object()

OperationInvT = TypeVar("OperationInvT", bound=Operation)

ConversionFunction = Callable[[OperationInvT, PatternRewriter], None]


def conversion(
    op_type: type[OperationInvT],
) -> Callable[[ConversionFunction], ConversionFunction]:
    """

    """
    def annot(
        func: ConversionFunction
    ) -> ConversionFunction:
        def impl(
            op: Operation, rewriter: PatternRewriter
        ):
            func(op, rewriter)

        setattr(impl, _CONVERSION_INPUT_TYPE, op_type)
        return impl

    return annot

@dataclass(frozen=True)
class ConversionPass(ABC, ModulePass):
    _conversion_patterns: dict[type[Operation], Callable[[Operation, PatternRewriter], None]] = field(default_factory=dict, init=False, repr=False, hash=False,compare=False)

    legal_ops: ClassVar[set[type[Operation]]]
    illegal_ops: ClassVar[set[type[Operation]]]

    mode: ClassVar[ConversionMode]

    def __post_init__(self):
        for field in self.__dict__.values():
            if _CONVERSION_INPUT_TYPE in field.__dir__():
                inp_t = getattr(_CONVERSION_INPUT_TYPE, field)
                if inp_t not in self.illegal_ops:
                    raise ValueError(f"OpType {inp_t} cannot be converted because it is not declared illegal!")
                self._conversion_patterns[inp_t] = field


    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        # rewrite the module
        PatternRewriteWalker(_ConversionPattern(self._conversion_patterns)).rewrite_module(op)

        # check that all ops were converted:
        for subop in op.walk():
            if type(subop) in self.illegal_ops:
                raise ConversionPassError(f"Illegal operation after conversion: {subop}")
            if type(subop) not in self.legal_ops and self.mode == ConversionMode.Full:
                raise ConversionPassError(f"Full conversion requires all operations to be explicitly legal, unknown op found: {subop}")


@dataclass
class _ConversionPattern(RewritePattern):
    """
    Rewrites all ops for which a conversion is known, used as part of the
    ConversionPass
    """
    conversions: dict[type[Operation], Callable[[Operation, PatternRewriter], None]]
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if type(op) in self.conversions:
            self.conversions[type(op)](op, rewriter)
