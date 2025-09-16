"""
Interfaces are a convenience wrapper around traits, providing logic on the operation
directly.

Operations inherit all the traits of their superclasses, allowing them to combine
behaviours via sublcassing.
This can be more convenient than adding the traits explicitly.
"""

import abc
from dataclasses import dataclass
from typing import cast

from xdsl.ir import Attribute, Operation
from xdsl.irdl import traits_def
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import ConstantLike, HasCanonicalizationPatternsTrait


@dataclass(frozen=True)
class _HasCanonicalizationPatternsInterfaceTrait(HasCanonicalizationPatternsTrait):
    """
    Gets the canonicalization patterns from the operation's implementation
    of `HasCanonicalizationPatternsInterface`.
    """

    def verify(self, op: Operation) -> None:
        return

    def get_patterns(
        self,
        op: type[Operation],
    ) -> tuple[RewritePattern, ...]:
        op = cast(type[HasCanonicalizationPatternsInterface], op)
        return op.get_canonicalization_patterns()


class HasCanonicalizationPatternsInterface(Operation, abc.ABC):
    """
    An operation subclassing this interface must implement the
    `get_canonicalization_patterns` method, which returns a tuple of patterns that
    canonicalize this operation.
    Wraps `CanonicalizationPatternsTrait`.
    """

    traits = traits_def(_HasCanonicalizationPatternsInterfaceTrait())

    @classmethod
    @abc.abstractmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        raise NotImplementedError()


class _ConstantLikeInterfaceTrait(ConstantLike):
    """
    Gets the constant value from the operation's implementation
    of `ConstantLikeInterface`.
    """

    def verify(self, op: Operation) -> None:
        return

    @classmethod
    def get_constant_value(cls, op: Operation) -> Attribute:
        op = cast(ConstantLikeInterface, op)
        return op.get_constant_value()


class ConstantLikeInterface(Operation, abc.ABC):
    """
    An operation subclassing this interface must implement the
    `get_constant_value` method, which returns the constant value of this operation.
    Wraps `ConstantLikeTrait`.
    """

    traits = traits_def(_ConstantLikeInterfaceTrait())

    @abc.abstractmethod
    def get_constant_value(self) -> Attribute:
        raise NotImplementedError()
