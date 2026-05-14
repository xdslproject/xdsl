"""
Hacky prototype pass: hardcode acc/upper/lower bound attributes on every
math.exp op in the module.

Intended as a fast way to exercise the exp→polynomial lowering end-to-end
before introducing proper pass parameters or front-end annotations.

The defaults below assume the model's math.exp ops appear inside a softmax
(`exp(x - max(x))`), so the argument is always <= 0. The lower bound is set
conservatively but tighter than the format's representable underflow, which
covers most NN softmax inputs in practice.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import math
from xdsl.dialects.builtin import Float64Type, FloatAttr, ModuleOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class AttachExpBoundsPass(ModulePass):
    """Attach hardcoded acc/upper/lower bound attributes to every math.exp op."""

    name = "attach-exp-bounds"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        f64 = Float64Type()
        for child in op.walk():
            if not isinstance(child, math.ExpOp):
                continue
            # Don't clobber attributes the user already set explicitly.
            if "acc_bound" not in child.attributes:
                child.attributes["acc_bound"] = FloatAttr(1e-4, f64)
            if "upper_bound" not in child.attributes:
                # softmax computes exp(x - max(x)); argument is always <= 0.
                child.attributes["upper_bound"] = FloatAttr(0.0, f64)
            if "lower_bound" not in child.attributes:
                # Generous default that covers most softmax inputs in practice.
                # Tighter than the format's representable underflow.
                child.attributes["lower_bound"] = FloatAttr(-50.0, f64)
