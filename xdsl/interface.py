from __future__ import annotations

import abc
from dataclasses import dataclass

from xdsl.ir import Operation
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait


@dataclass(frozen=True)
class _HasCanonicalizationPatternsInterfaceTrait(HasCanonicalizationPatternsTrait):
    a: type[HasCanonicalizationPatternsInterface]

    def get_canonicalization_patterns(self) -> tuple[RewritePattern, ...]:
        return self.a.get_canonicalization_patterns()


class HasCanonicalizationPatternsInterface(Operation):
    """
    Provides the rewrite passes to canonicalize an operation.

    Each rewrite pattern must have the trait's op as root.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.traits.add_trait(_HasCanonicalizationPatternsInterfaceTrait(cls))

    @abc.abstractmethod
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        raise NotImplementedError()
