#!/usr/bin/env python3
"""Benchmark the time to check interface and trait properties."""

from xdsl.dialects.gpu import TerminatorOp
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_successor_def,
    traits_def,
)
from xdsl.traits import IsTerminator, NoTerminator


@irdl_op_definition
class IsTerminatorOp(IRDLOperation):
    """
    An operation that provides the IsTerminator trait.
    """

    name = "test.is_terminator"

    successor = opt_successor_def()

    traits = traits_def(IsTerminator())


IS_TERMINATOR_OP = TerminatorOp()


def time_extensibility__interface_check() -> None:
    """."""
    assert isinstance(IS_TERMINATOR_OP, TerminatorOp)


def time_extensibility__trait_check() -> None:
    """."""
    assert IS_TERMINATOR_OP.has_trait(IsTerminator)
    assert not IS_TERMINATOR_OP.has_trait(NoTerminator)


if __name__ == "__main__":
    from collections.abc import Callable

    from utils import profile

    BENCHMARKS: dict[str, Callable[[], None]] = {}
    profile(BENCHMARKS)
