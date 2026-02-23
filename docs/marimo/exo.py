import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Exo-Style Scheduling in xDSL

    Applying 2D tiling on matrix multiplication.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.parser import Parser
    from xdsl.context import Context
    from xdsl.dialects import arith, func, test, scf, builtin, memref
    from xdsl.transforms.scf_for_loop_range_folding import ScfForLoopRangeFoldingPass
    from xdsl.transforms.canonicalize import CanonicalizePass
    return (
        CanonicalizePass,
        Context,
        Parser,
        ScfForLoopRangeFoldingPass,
        arith,
        builtin,
        func,
        memref,
        mo,
        scf,
        test,
        xmo,
    )


@app.cell
def _(Context, arith, builtin, func, memref, scf, test):
    ctx = Context()

    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(test.Test)
    ctx.load_dialect(memref.MemRef)
    return (ctx,)


@app.cell
def _(Parser, ctx, xmo):
    # Input matmul function

    input_str = """
      func.func @matmul(%A : memref<512x512xf32>, %B : memref<512x512xf32>, %C : memref<512x512xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c512 = arith.constant 512 : index
        scf.for %i = %c0 to %c512 step %c1 {
          scf.for %j = %c0 to %c512 step %c1 {
            scf.for %k = %c0 to %c512 step %c1 {
              %a = memref.load %A[%i, %k] : memref<512x512xf32>
              %b = memref.load %B[%k, %j] : memref<512x512xf32>
              %acc_old = memref.load %C[%i, %j] : memref<512x512xf32>
              %prod = arith.mulf %a, %b : f32
              %acc_new = arith.addf %acc_old, %prod : f32
              memref.store %acc_new, %C[%i, %j] : memref<512x512xf32>
            }
          }
        }
        func.return
      }
    """

    input_module = Parser(ctx, input_str).parse_module()
    input_module.verify()
    xmo.module_html(input_module)
    return (input_module,)


@app.cell
def _(Operation, input_module, scf):
    # Exo-like "find" function for getting a reference to an operation by pattern matching

    from xdsl.dialects.builtin import ModuleOp

    from typing_extensions import TypeVar

    _T = TypeVar("_T", bound=Operation)

    def find(module: ModuleOp, pattern: type[_T]) -> tuple[_T, ...]:
        return tuple(op for op in module.walk() if isinstance(op, pattern))

    find(input_module, scf.ForOp)
    return ModuleOp, find


@app.cell
def _():
    from typing import cast

    from xdsl.printer import Printer
    from xdsl.ir import OpResult, Operation, Region, Block
    from xdsl.builder import InsertPoint
    from xdsl.pattern_rewriter import (
        Rewriter,
        op_type_rewrite_pattern,
    )
    from dataclasses import dataclass
    return Block, InsertPoint, Operation, Printer, Region, Rewriter


@app.cell
def _(
    Block,
    InsertPoint,
    ModuleOp,
    Region,
    Rewriter,
    arith,
    builtin,
    func,
    scf,
):
    # Implementation of the "split" primitive. The function signature mirrors Exo,
    # taking a module (proc), a cursor, and a div_const. Module is mutable in MLIR so that's different.
    # This split impelementation is at least sound

    # TODO: cursor is just an Operation for now. We need something better when we support forwarding
    def split(module : ModuleOp, cursor: scf.ForOp, div_const: int) -> tuple[scf.ForOp, scf.ForOp]:
        r = Rewriter()

        assert isinstance(cursor, scf.ForOp)

        # check the lower bound
        assert isinstance(cursor.lb.owner, arith.ConstantOp)
        assert cursor.lb.owner.value.value.data == 0

        # check the upper bound
        assert isinstance(cursor.ub.owner, arith.ConstantOp)
        # upper bound should be perfectly divisible by div_const
        assert cursor.ub.owner.value.value.data % div_const == 0

        # check the step
        assert cursor.step.owner.value.value.data == 1

        parent_func = cursor
        while not isinstance(parent_func, func.FuncOp):
            parent_func = parent_func.parent_op()

        res_names = tuple(r.name_hint for r in cursor.results)

        step_op = arith.ConstantOp(builtin.IntegerAttr(div_const, cursor.step.type))
        r.insert_op(step_op, insertion_point=InsertPoint.at_start(parent_func.body.block))
        step_op.result.name_hint = f"c{div_const}"

        inner_body = r.move_region_contents_to_new_regions(cursor.body) # this is region
        ii, *iter_args_i = inner_body.block.args

        outer_body = Region(Block(arg_types=(ii.type, *(val.type for val in iter_args_i))))
        io, *iter_args_o = outer_body.block.args

        inner_loop = scf.ForOp(cursor.lb, step_op.result, cursor.step, iter_args_o, inner_body)
        for old, new_i_res in zip(res_names, inner_loop.results):
            if old is not None:
                new_i_res.name_hint = old + "_i"

        r.insert_op(inner_loop, InsertPoint.at_start(outer_body.block))

        outer_loop = scf.ForOp(cursor.lb, cursor.ub, step_op, cursor.iter_args, outer_body)
        new_i = arith.AddiOp(ii, io)
        r.insert_op(new_i, insertion_point=InsertPoint.at_start(inner_body.block))
        if ii.name_hint is not None:
            new_i.result.name_hint = ii.name_hint
        ii.replace_by_if(new_i.result, lambda val : val.operation != new_i)

        r.replace_op(cursor, outer_loop)
        for old, new_o_res in zip(res_names, outer_loop.results):
            if old is not None:
                new_o_res.name_hint = old + "_o"

        for (outer_arg, inner_arg) in zip(outer_body.block.args, inner_body.block.args, strict=True):
            if inner_arg.name_hint is not None:
                outer_arg.name_hint = inner_arg.name_hint + "_o"
                inner_arg.name_hint += "_i"

        return outer_loop, inner_loop
    return (split,)


@app.cell
def _(Block, ModuleOp, Region, Rewriter, arith, scf):
    # Implementation of "reorder_loops"
    # TODO: reorder_loops does not check the commutativity of the body of the K loop. It needs to be asserted from the user.
    def reorder_loops(module : ModuleOp, o_loop: scf.ForOp, i_loop: scf.ForOp) -> tuple[scf.ForOp, scf.ForOp]:
        r = Rewriter()

        # body of the outer loop should be size 1. Loops must be perfectly nested
        assert len(o_loop.body.block.ops) == 2
        assert i_loop.parent_op() is o_loop

        # check that inner loop bounds does not depend on the outer loop iteration variable
        # actually we can just check if they're constant or not
        assert isinstance(i_loop.ub.owner, arith.ConstantOp)
        assert isinstance(i_loop.lb.owner, arith.ConstantOp)

        new_body = r.move_region_contents_to_new_regions(i_loop.body)
        new_i_loop = scf.ForOp(o_loop.lb, o_loop.ub, o_loop.step, (), new_body)
        outer_body  = Region(Block([new_i_loop], arg_types=(o_loop.body.block.args[0].type,)))
        new_o_loop = scf.ForOp(i_loop.lb, i_loop.ub, i_loop.step, (), outer_body)
        r.replace_op(o_loop, new_o_loop)

        for (old_inner, new_outer) in zip(new_i_loop.body.block.args, new_o_loop.body.block.args, strict=True):
            new_outer.name_hint = old_inner.name_hint

        for (old_outer, new_inner) in zip(o_loop.body.block.args, new_i_loop.body.block.args, strict=True):
            new_inner.name_hint = old_outer.name_hint

        return new_o_loop, new_i_loop
    return (reorder_loops,)


@app.cell
def _(
    CanonicalizePass,
    Printer,
    ScfForLoopRangeFoldingPass,
    ctx,
    find,
    input_module,
    reorder_loops,
    scf,
    split,
):
    # Tile 2D rewrite

    print("IR before tiling:")
    Printer().print_op(input_module)
    _module =input_module.clone()

    # ---- Scheduling code begin -----
    cursors = find(_module, scf.ForOp) # this should be loops
    io, ii = split(_module, cursors[0], 16) # split the loop cursors[0] is pointing
    jo, ji = split(_module, cursors[1], 8)
    ScfForLoopRangeFoldingPass().apply(ctx, _module) # hack
    CanonicalizePass().apply(ctx, _module) # hack
    reorder_loops(_module, ii, jo)
    # ---- Scheduling code end -----

    print("\n")
    print("IR after tiling:")
    Printer().print_op(_module)
    return


if __name__ == "__main__":
    app.run()
