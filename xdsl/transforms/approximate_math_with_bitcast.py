import math as pmath
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin, math
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.utils.hints import isa


@dataclass
class MakeBase2(RewritePattern):
    log: bool
    exp: bool

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if len(op.results) != 1:
            return

        t = op.results[0].type
        if not isa(t, builtin.AnyFloat):
            return

        match op:
            # rewrite ln(A) -> log2(A) * ln(2)
            case math.LogOp() if self.log:
                ln2 = builtin.FloatAttr(pmath.log(2), t)

                rewriter.replace_matched_op(
                    [
                        c := arith.ConstantOp(ln2),
                        newlog := math.Log2Op(op.operand, op.fastmath),
                        mul := arith.MulfOp(c, newlog, op.fastmath),
                    ],
                    mul.results,
                )
            # rewrite eexp(%a) to exp2(%a * log2(e))
            case math.ExpOp() if self.exp:
                log2e = builtin.FloatAttr(pmath.log2(pmath.e), t)
                rewriter.replace_matched_op(
                    [
                        c := arith.ConstantOp(log2e),
                        inner := arith.MulfOp(c, op.operand, op.fastmath),
                        e := math.Exp2Op(inner, op.fastmath),
                    ],
                    e.results,
                )
            # TODO: math.powf
            # TODO: math.log10
            # TODO: math.log1p
            # TODO: math.fpowi?
            case _:
                pass


LBs = {
    builtin.f16: (2**10, 14),
    builtin.f32: (2**23, 127),
    builtin.f64: (2**52, 1023),
}


@dataclass
class MakeApprox(RewritePattern):
    log: bool
    exp: bool

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if len(op.results) == 0:
            return

        t = op.results[0].type
        if not isa(t, builtin.AnyFloat):
            return

        L, B = LBs[t] if t in LBs else (1, 1)

        int_t = builtin.IntegerType(t.bitwidth)

        match op:
            # log2(%a) -> fp(L * (B - eps + A))
            case math.Log2Op(operand=x) if self.log:
                rewriter.replace_matched_op(
                    [
                        a := arith.ConstantOp(builtin.FloatAttr(L, t)),
                        b := arith.ConstantOp(builtin.FloatAttr(L * (B - 0.045), t)),
                        ax := arith.MulfOp(a, x, op.fastmath),
                        axpb := arith.AddfOp(b, ax, op.fastmath),
                        asint := arith.FPToSIOp(axpb, int_t),
                        res := arith.BitcastOp(asint, t),
                    ],
                    res.results,
                )
            case math.Exp2Op(operand=x) if self.exp:
                # 2^%x -> int(%x) * 1/L - B + eps
                rewriter.replace_matched_op(
                    [
                        a := arith.ConstantOp(builtin.FloatAttr(1 / L, t)),
                        b := arith.ConstantOp(builtin.FloatAttr(-B + 0.045, t)),
                        xi := arith.BitcastOp(x, int_t),
                        xif := arith.SIToFPOp(xi, t),
                        ax := arith.MulfOp(a, xif, op.fastmath),
                        axpb := arith.AddfOp(b, ax, op.fastmath),
                    ],
                    axpb.results,
                )
            case _:
                pass


@dataclass(frozen=True)
class BitcastApproximation(ModulePass):
    r"""
    This pass applies approximations for some math operations (currently log and exp)
    and converts them to bitcasting-based approximations.

    These are intended for environments that don't need high accuracy, and do not have
    specialized hardware support for expf and log in hardware.

    It makes use of the fact that IEEE floating-point numbers are encoded as three base-2
    numbers:

        s eeeeeeee mmmmmmmmmmmmmmmmmmmmmmm

    With the final floating-point number being $$x = (-1)^s * 2^{e-B} * (1 + m/L)$$.
    Hence, $$e$$ encodes the $$\lfloor log2(x)\rfloor$$ of x, and can be accessed by
    bit-casting and left-shifting.

    The following approximations are enabled through this:

    1) $$\log_2(x) \approx bc-int(x)/L - B + \varepsilon$$
    2) $$2^x       \approx bc-float(L(B-\varepsilon + x))$$

    With $$\varepsilon$$ being a tunable constant that we initialize to 0.045 for simplicity

    This pass first applies some rewrites to convert suitable arithmetic into base-2 format,
    and then applies the above approximations.

    Inidividual rewrites can be controlled via pass arguments.
    """

    name = "approximate-math-with-bitcast"

    log: bool = True
    exp: bool = True

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    MakeBase2(self.log, self.exp),
                    MakeApprox(self.log, self.exp),
                ]
            )
        ).rewrite_module(op)
