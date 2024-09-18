from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    SymbolRefAttr,
)
from xdsl.dialects.csl import csl, csl_stencil, csl_wrapper
from xdsl.ir import (
    Block,
    Operation,
    OpResult,
    Region,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class HandleCslStencilApplyAsyncCF(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        if op.next_op and isinstance(op.next_op, csl.ReturnOp):
            return
        if (
            op.next_op
            and isinstance(call_op := op.next_op, csl.CallOp)
            and op.next_op.next_op
            and isinstance(op.next_op.next_op, csl.ReturnOp)
        ):
            terminator = op.post_process.block.last_op
            assert isinstance(terminator, csl_stencil.YieldOp)
            rewriter.insert_op(call_op.clone(), InsertPoint.before(terminator))
            rewriter.erase_op(call_op)
            return
        # todo there is more code here, split the code into a new func and call this


@dataclass()
class ConvertForLoopToCallGraphPass(RewritePattern):
    """
    Translates for-loop to csl functions, to handle async communications in the loop body.
    Splits the body of the enclosing function into: pre_block, loop, post_block.
    Further splits `loop` into for_cond, for_body, for_inc.
    Loop var and iter_args are promoted to module-level csl vars.

    Limitations:
      * Loop can only yield its input iter_args (in any order)
      * Loop bounds and step must be arith.constant values
    """

    counter: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        if not self._is_inside_wrapper_outside_apply(op):
            return

        parent_func = op.parent_op()
        while parent_func is not None and not isinstance(parent_func, csl.FuncOp):
            parent_func = parent_func.parent_op()
        if not parent_func:
            return

        # limitation: can yield iter_args in any order, but they cannot be modified in the loop body
        terminator = op.body.block.last_op
        assert isinstance(terminator, scf.Yield)
        assert all(
            arg in op.body.block.args for arg in terminator.arguments
        ), "Can only yield unmodified iter_args (in any order)"

        # limitation: currently only loops built from arith.constant are supported
        assert isinstance(op.lb, OpResult)
        assert isinstance(op.ub, OpResult)
        assert isinstance(op.step, OpResult)
        assert isinstance(op.lb.op, arith.Constant)
        assert isinstance(op.ub.op, arith.Constant)
        assert isinstance(op.step.op, arith.Constant)
        assert isa(op.lb.op.value, IntegerAttr[IndexType])
        assert isa(op.ub.op.value, IntegerAttr[IndexType])
        assert isa(op.step.op.value, IntegerAttr[IndexType])

        # limitation: all iter_args must be memrefs (stencil buffers) and have the same data type
        assert isa(op.iter_args[0].type, MemRefType[csl.ZerosOp.T])
        element_type = op.iter_args[0].type.get_element_type()
        assert all(
            isa(a.type, MemRefType[csl.ZerosOp.T])
            and element_type == a.type.get_element_type()
            for a in op.iter_args
        )

        no_params = FunctionType.from_lists([], [])

        pre_block = op.parent_block()
        if pre_block is None:
            return
        post_block = pre_block.split_before(op)

        post_func = csl.FuncOp(f"for_post{self.counter}", no_params)
        rewriter.inline_block(post_block, InsertPoint.at_start(post_func.body.block))

        cond_func = csl.TaskOp(
            f"for_cond{self.counter}",
            no_params,
            task_kind=csl.TaskKind.LOCAL,
            id=self.counter + 1,
        )
        body_func = csl.FuncOp(f"for_body{self.counter}", no_params)
        inc_func = csl.FuncOp(f"for_inc{self.counter}", no_params)

        # create csl.vars for loop var and iter_args outside the parent func
        rewriter.insert_op(
            iv := csl.VariableOp.from_value(IntegerAttr(op.lb.op.value.value, 16)),
            InsertPoint.before(parent_func),
        )
        iter_vars = [csl.VariableOp.from_type(arg_t) for arg_t in op.iter_args.types]
        rewriter.insert_op(iter_vars, InsertPoint.before(parent_func))

        # parent func (pre loop): setup iter vars and activate cond_func
        with ImplicitBuilder(pre_block):
            for dst, src in zip(iter_vars, op.iter_args):
                csl.StoreVarOp(dst, src)
            csl.CallOp(SymbolRefAttr(cond_func.sym_name))
            csl.ReturnOp()

        # for-loop condition func
        with ImplicitBuilder(cond_func.body.block):
            ub = arith.Constant.from_int_and_width(op.ub.op.value.value, 16)
            iv_load = csl.LoadVarOp(iv)
            cond = arith.Cmpi(iv_load, ub, "slt")
            branch = scf.If(cond, [], Region(Block()), Region(Block()))
            with ImplicitBuilder(branch.true_region):
                csl.CallOp(SymbolRefAttr(body_func.sym_name))
            with ImplicitBuilder(branch.false_region):
                csl.CallOp(SymbolRefAttr(post_func.sym_name))
            csl.ReturnOp()

        # for-loop inc func
        with ImplicitBuilder(inc_func.body.block):
            step = arith.Constant.from_int_and_width(op.step.op.value.value, 16)
            iv_load = csl.LoadVarOp(iv)
            stepped = arith.Addi(iv_load, step)
            csl.StoreVarOp(iv, stepped)
            load_vars = [csl.LoadVarOp(v) for v in iter_vars]
            for iter_var, yielded_var in zip(iter_vars, terminator.arguments):
                idx = (
                    op.body.block.args.index(yielded_var) - 1
                )  # subtract 1 as the first block arg is the loop var
                csl.StoreVarOp(iter_var, load_vars[idx])
            csl.CallOp(SymbolRefAttr(cond_func.sym_name))
            csl.ReturnOp()

        # for-loop body func
        with ImplicitBuilder(body_func.body.block):
            body_vars = [csl.LoadVarOp(var) for var in [iv, *iter_vars]]
        rewriter.inline_block(
            op.body.block,
            InsertPoint.at_end(body_func.body.block),
            [v.res for v in body_vars],
        )
        rewriter.insert_op(
            csl.CallOp(SymbolRefAttr(inc_func.sym_name)), InsertPoint.before(terminator)
        )
        rewriter.replace_op(terminator, csl.ReturnOp())

        # place funcs and erase now-empty for-loop
        rewriter.insert_op(
            [cond_func, body_func, inc_func, post_func], InsertPoint.after(parent_func)
        )
        rewriter.erase_matched_op()

    @staticmethod
    def _is_inside_wrapper_outside_apply(op: Operation):
        """Returns if the op is inside `csl_wrapper.module` and contains a `csl_stencil.apply`."""
        is_inside_wrapper = False
        is_inside_apply = False
        has_apply_inside = False

        parent_op = op.parent_op()
        while parent_op:
            if isinstance(parent_op, csl_wrapper.ModuleOp):
                is_inside_wrapper = True
            elif isinstance(parent_op, csl_stencil.ApplyOp):
                is_inside_apply = True
            parent_op = parent_op.parent_op()

        for child_op in op.walk():
            if isinstance(child_op, csl_stencil.ApplyOp):
                has_apply_inside = True
                break

        return is_inside_wrapper and not is_inside_apply and has_apply_inside


@dataclass(frozen=True)
class CslStencilHandleAsyncControlFlow(ModulePass):
    """ """

    name = "csl-stencil-handle-async-flow"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertForLoopToCallGraphPass(0),
                    HandleCslStencilApplyAsyncCF(),
                ]
            ),
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
