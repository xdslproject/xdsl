from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    SymbolRefAttr,
    i32,
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
from xdsl.utils.exceptions import PassFailedException
from xdsl.utils.hints import isa


@dataclass()
class HandleCslStencilApplyAsyncCF(RewritePattern):
    """
    Ensure that the async csl_stencil.apply op is last in its function.
    Any code following an apply is split off into a separate function.
    Control flow proceeds from the apply's second callback to the split-off function.
    """

    counter: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        # invalid case
        if not op.next_op:
            return

        # case 1: apply is last in function, nothing to do
        if isinstance(op.next_op, csl.ReturnOp):
            return

        terminator = op.done_exchange.block.last_op
        assert isinstance(terminator, csl_stencil.YieldOp)

        # case 2: apply is followed by call_op and return - move call_op to second callback of apply
        if (
            isinstance(call_op := op.next_op, csl.CallOp)
            and op.next_op.next_op
            and isinstance(op.next_op.next_op, csl.ReturnOp)
        ):
            rewriter.insert_op(call_op.clone(), InsertPoint.before(terminator))
            rewriter.erase_op(call_op)
            return

        parent_func = op.parent_op()
        while parent_func is not None and not isinstance(parent_func, csl.FuncOp):
            parent_func = parent_func.parent_op()
        if not parent_func:
            return

        # case 3: apply is followed by other code, split it off into a different func, call it from second callback of apply
        assert (parent_block := op.parent_block()) is not None
        next_block = parent_block.split_before(op.next_op)
        rewriter.insert_op(csl.ReturnOp(), InsertPoint.after(op))
        next_func = csl.FuncOp(f"step{self.counter}", FunctionType.from_lists([], []))
        self.counter += 1
        rewriter.inline_block(next_block, InsertPoint.at_start(next_func.body.block))
        rewriter.insert_op(
            csl.CallOp(SymbolRefAttr(next_func.sym_name)),
            InsertPoint.before(terminator),
        )
        rewriter.insert_op(next_func, InsertPoint.after(parent_func))


@dataclass()
class ConvertForLoopToCallGraphPass(RewritePattern):
    """
    Translates for-loop to csl functions, to handle async communications in the loop body.
    Splits the body of the enclosing function into: pre_block, loop, post_block.
    Further splits `loop` into for_cond, for_body, for_inc.
    These can in theory all be functions, yet we setup `for_cond` as a local task
    to avoid potentially large in-memory call stacks for long-running loops.
    Loop var and iter_args are promoted to module-level csl vars.

    Limitations:
      * Loop can only yield its input iter_args (in any order)
      * Loop bounds and step must be arith.constant values
      * Iter args must currently be stencil buffers (memrefs) of the same data type
    """

    counter: int

    task_ids: list[int]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter, /):
        if not self._is_inside_wrapper_outside_apply(op):
            return

        parent_func = op.parent_op()
        while parent_func is not None and not isinstance(parent_func, csl.FuncOp):
            parent_func = parent_func.parent_op()
        if not parent_func:
            return

        # limitation: can yield iter_args in any order, but they cannot be modified in the loop body
        terminator = op.body.block.last_op
        assert isinstance(terminator, scf.YieldOp)
        assert all(arg in op.body.block.args for arg in terminator.arguments), (
            "Can only yield unmodified iter_args (in any order)"
        )

        # limitation: currently only loops built from arith.constant are supported
        assert isinstance(op.lb, OpResult)
        assert isinstance(op.ub, OpResult)
        assert isinstance(op.step, OpResult)
        assert isinstance(op.lb.op, arith.ConstantOp)
        assert isinstance(op.ub.op, arith.ConstantOp)
        assert isinstance(op.step.op, arith.ConstantOp)
        assert isa(op.lb.op.value, IntegerAttr[IndexType])
        assert isa(op.ub.op.value, IntegerAttr[IndexType])
        assert isa(op.step.op.value, IntegerAttr[IndexType])

        # limitation: all iter_args must be memrefs (stencil buffers) and have the same data type
        if op.iter_args:
            assert isa(op.iter_args[0].type, MemRefType)
            element_type = op.iter_args[0].type.get_element_type()
            assert all(
                isa(a.type, MemRefType) and element_type == a.type.get_element_type()
                for a in op.iter_args
            )

        no_params = FunctionType.from_lists([], [])
        if self.task_ids:
            cond_task_id = self.task_ids.pop(0)
        else:
            raise PassFailedException(
                "Insufficient number of task IDs supplied, please provide further IDs to be used."
            )

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
            id=cond_task_id,
        )
        body_func = csl.FuncOp(f"for_body{self.counter}", no_params)
        inc_func = csl.FuncOp(f"for_inc{self.counter}", no_params)

        # create csl.vars for loop var and iter_args outside the parent func
        rewriter.insert_op(
            iv := csl.VariableOp.from_value(IntegerAttr(op.lb.op.value.value, i32)),
            InsertPoint.before(parent_func),
        )
        iter_vars = [csl.VariableOp.from_type(arg_t) for arg_t in op.iter_args.types]
        rewriter.insert_op(iter_vars, InsertPoint.before(parent_func))

        iv.res.name_hint = "iteration"
        for i, v in enumerate(iter_vars):
            v.res.name_hint = f"var{i}"

        # parent func (pre loop): setup iter vars and activate cond_func
        with ImplicitBuilder(pre_block):
            for dst, src in zip(iter_vars, op.iter_args):
                csl.StoreVarOp(dst, src)
            csl.ActivateOp(cond_task_id, csl.TaskKind.LOCAL)
            csl.ReturnOp()

        # for-loop condition func
        with ImplicitBuilder(cond_func.body.block):
            ub = arith.ConstantOp.from_int_and_width(op.ub.op.value.value, i32)
            iv_load = csl.LoadVarOp(iv)
            iv_load.res.name_hint = f"{iv.res.name_hint}_cond"
            cond = arith.CmpiOp(iv_load, ub, "slt")
            branch = scf.IfOp(cond, [], Region(Block()), Region(Block()))
            with ImplicitBuilder(branch.true_region):
                csl.CallOp(SymbolRefAttr(body_func.sym_name))
                scf.YieldOp()
            with ImplicitBuilder(branch.false_region):
                csl.CallOp(SymbolRefAttr(post_func.sym_name))
                scf.YieldOp()
            csl.ReturnOp()

        # for-loop inc func
        with ImplicitBuilder(inc_func.body.block):
            step = arith.ConstantOp.from_int_and_width(op.step.op.value.value, i32)
            iv_load = csl.LoadVarOp(iv)
            iv_load.res.name_hint = f"{iv.res.name_hint}_inc"
            stepped = arith.AddiOp(iv_load, step)
            csl.StoreVarOp(iv, stepped)

            # pre-load iter_vars and store them in the order specified in scf.yield
            load_vars = [csl.LoadVarOp(v) for v in iter_vars]
            for v in load_vars:
                v.res.name_hint = f"{v.var.name_hint}_inc"

            # for out-of-order yields, store yielded var to iter_var
            for iter_var, yielded_var in zip(iter_vars, terminator.arguments):
                # `idx` is the original index of the yielded var, subtract 1 as the first block arg is the loop var
                idx = op.body.block.args.index(yielded_var) - 1
                csl.StoreVarOp(iter_var, load_vars[idx])
            csl.ActivateOp(cond_task_id, csl.TaskKind.LOCAL)
            csl.ReturnOp()

        # for-loop body func
        with ImplicitBuilder(body_func.body.block):
            body_vars = [csl.LoadVarOp(var) for var in [iv, *iter_vars]]
            for v in body_vars:
                v.res.name_hint = f"{v.var.name_hint}_bdy"
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
        rewriter.erase_op(op)

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
class CopyArithConstants(RewritePattern):
    """ """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter, /):
        if not (parent_func := self._get_enclosing_function(op)):
            return
        for use in list(op.result.uses):
            use_func = self._get_enclosing_function(use.operation)
            if use_func != parent_func:
                rewriter.insert_op(cln := op.clone(), InsertPoint.before(use.operation))
                op.result.replace_by_if(cln.result, lambda x: x == use)

    @staticmethod
    def _get_enclosing_function(op: Operation) -> csl.FuncOp | None:
        parent = op.parent_op()
        while parent and not isinstance(parent, csl.FuncOp):
            parent = parent.parent_op()
        return parent


@dataclass(frozen=True)
class CslStencilHandleAsyncControlFlow(ModulePass):
    """
    Handles the async control flow of csl_stencil.apply and any enclosing loops
    by translating control flow into a csl.func call graph.
    """

    name = "csl-stencil-handle-async-flow"

    task_ids: tuple[int, ...]
    """
    Available task IDs that this pass is free to allocate.
    """

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertForLoopToCallGraphPass(0, list(self.task_ids)),
                    HandleCslStencilApplyAsyncCF(0),
                ]
            ),
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
        PatternRewriteWalker(
            CopyArithConstants(), apply_recursively=False
        ).rewrite_module(op)
