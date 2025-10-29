from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, func, llvm
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.ir import Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable


class ArgNamesToArgAttrsPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        if op.is_declaration or not any(arg.name_hint for arg in op.args):
            return

        arg_attrs = (
            op.arg_attrs.data
            if op.arg_attrs is not None
            else ((DictionaryAttr({}),) * len(op.args))
        )

        new_arg_attrs = ArrayAttr(
            DictionaryAttr(
                arg_attr.data.set("llvm.name", StringAttr(arg.name_hint))
                if arg.name_hint and "llvm.name" not in arg_attr.data
                else arg_attr.data
            )
            for arg, arg_attr in zip(op.args, arg_attrs, strict=True)
        )

        if new_arg_attrs != op.arg_attrs:
            op.arg_attrs = new_arg_attrs
            rewriter.has_done_action = True


TIMER_START = "timer_start"
TIMER_END = "timer_end"


@dataclass
class AddBenchTimersPattern(RewritePattern):
    start_func_t: func.FunctionType
    end_func_t: func.FunctionType

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        if (
            not (top_level := op.parent_op())
            or not isinstance(top_level, builtin.ModuleOp)
            or top_level.parent
        ):
            return

        ptr = op.body.block.insert_arg(llvm.LLVMPointerType(), len(op.args))
        start_call = func.CallOp(TIMER_START, [], tuple(self.start_func_t.outputs))
        end_call = func.CallOp(
            TIMER_END, start_call.res, tuple(self.end_func_t.outputs)
        )
        store_time = llvm.StoreOp(end_call.res[0], ptr)

        ptr.name_hint = "timers"
        start_call.res[0].name_hint = "timestamp"
        end_call.res[0].name_hint = "timediff"

        assert op.body.block.last_op
        rewriter.insert_op(start_call, InsertPoint.at_start(op.body.block))
        rewriter.insert_op(
            [end_call, store_time], InsertPoint.before(op.body.block.last_op)
        )
        op.update_function_type()


class TestAddBenchTimersToTopLevelFunctions(ModulePass):
    """
    Adds timers to top-level functions, by adding `timer_start() -> f64` and `timer_end(f64) -> f64`
    to the start and end of each module-level function. The time is stored in an `llvm.ptr` passed in
    as a function arg.
    """

    name = "test-add-timers-to-top-level-funcs"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if SymbolTable.lookup_symbol(op, TIMER_START) or SymbolTable.lookup_symbol(
            op, TIMER_END
        ):
            return

        start_func_t = func.FunctionType.from_lists([], [builtin.f64])
        end_func_t = func.FunctionType.from_lists([builtin.f64], [builtin.f64])
        start_func = func.FuncOp(TIMER_START, start_func_t, Region([]), "private")
        end_func = func.FuncOp(TIMER_END, end_func_t, Region([]), "private")

        PatternRewriteWalker(
            AddBenchTimersPattern(start_func_t, end_func_t), apply_recursively=False
        ).rewrite_module(op)

        SymbolTable.insert_or_update(op, start_func)
        SymbolTable.insert_or_update(op, end_func)


class FunctionPersistArgNamesPass(ModulePass):
    """
    Persists func.func arg name hints to arg_attrs.

    Such that, for instance
        `func.func @my_func(%arg_name : i32) -> ...`
    becomes
        `func.func @my_func(%arg_name : i32 {"llvm.name" = "arg_name"}) -> ...
    """

    name = "function-persist-arg-names"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ArgNamesToArgAttrsPass(), apply_recursively=False
        ).rewrite_module(op)
