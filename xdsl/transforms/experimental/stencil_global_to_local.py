from dataclasses import dataclass
from typing import TypeVar
from abc import ABC, abstractmethod

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import MLContext
from xdsl.irdl import Attribute
from xdsl.dialects import builtin
from xdsl.dialects.experimental import stencil

_T = TypeVar('_T', bound=Attribute)


@dataclass
class SlicingStrategy(ABC):
    @abstractmethod
    def calc_resize(self, shape: tuple[int]) -> tuple[int]:
        raise NotImplementedError("SlicingStrategy must implement calc_resize!")


@dataclass
class HorizontalSlices2D(SlicingStrategy):
    slices: int

    def __post_init__(self):
        assert self.slices > 1, "must slice into at least two pieces!"

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        # slice on the last dimension only
        assert len(shape) == 2, "HorizontalSlices2D only works on 2d fields!"
        assert shape[1] % self.slices == 0, "HorizontalSlices2D expects second dim to be divisble by number of slices!"

        return shape[0], shape[1] // self.slices


@dataclass
class ChangeStoreOpSizes(RewritePattern):
    strategy: SlicingStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        assert all(integer_attr.value.data == 0 for integer_attr in op.lb.array.data), "lb must be 0"
        shape: tuple[int, ...] = tuple((integer_attr.value.data for integer_attr in op.ub.array.data))
        new_shape = self.strategy.calc_resize(shape)
        op.ub = stencil.IndexAttr.get(*new_shape)


@dataclass
class AddHaloExchangeOps(RewritePattern):
    """
    This rewrite adds a `stencil.halo_exchange` op after each `stencil.apply` call
    """
    strategy: SlicingStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.LoadOp, rewriter: PatternRewriter, /):
        swap_op = stencil.HaloSwapOp.get(op.field)
        rewriter.insert_op_before_matched_op(swap_op)


def global_stencil_to_local_stencil_2d_horizontal(ctx: MLContext | None, module: builtin.ModuleOp):
    strategy = HorizontalSlices2D(2)

    gpra = GreedyRewritePatternApplier([
        ChangeStoreOpSizes(strategy),
        AddHaloExchangeOps(strategy)
    ])

    PatternRewriteWalker(gpra, apply_recursively=False).rewrite_module(module)

