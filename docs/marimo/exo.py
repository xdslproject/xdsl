import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exo-Style Scheduling in xDSL

    This notebook describes how to implement schedules similarly to [Exo](github.com/exo-lang/exo), a Python-native framework for linear algebra kernel schedules.
    Exo divides the process of kernel implementation into two steps, first defining the kernel in a form that is easy to reason about and debug, and then applying a series of semantic-preserving transformations written in Python.
    This notebook will concentrate on the second part, but without the logic to validate that the transformations are semantic-preserving.

    We will build our way up to a transformation called 2D tiling.
    Tiling is a transformation is popularly applied to linear algebra kernels accessing memory in different n-dimensional arrays, as a means of optimising for cache locality.
    Here is a simple matrix multiplication kernel in pure Python before and after this transformation:

    ```py
    # A, B, and C assumed to be MxK, KxN, and MxN-dimension 2D arrays

    def before(A, B, C):
      for m in range(M):
        for n in range(N):
          for k in range(K):
            C[m, n] += A[m, k] + B[k, n]

    # Tiled in M and N by 16
    def after(A, B, C):
      for m_outer in range(0, M, 16):
        for n_outer in range(0, N, 16):
          for m in range(m_outer, m_outer + 16):
            for n in range(n_outer, n_outer + 16):
              for k in range(K):
                C[m, n] += A[m, k] + B[k, n]
    ```

    The first function computes each element of the output matrix by doing a dot product of the corresponding row and column of the input matrices.
    CPU caches typically store more than one element in a line, meaning that after loading an address in memory, a subsequent read of a nearby address in memory should be much faster due to already being saved in the cache, avoiding a part of the roundtrip.
    If the matrices have row-major layout, meaning the elements in rows are contiguous, fetching the elements of the row of A are expected to be faster, as these are laid out close in memory, and the row is reused to compute all of the elements of the first row of C before fetching the next row.
    In contrast, B will likely slow to access, as each subsequent read will be from a different row, and hence probably from a different cache line.
    If the size of B is larger than that of the cache, by the time that the next row of C is computed, it's likely that cached elements of B will have been evicted to make space for other values.

    After tiling, the access pattern has changed, even if the output result will be the same.
    Instead of computing the elements of C row by row, they will now be computed by 16x16 tile.
    It's more likely that the Kx16 slice of B will fit in the cache, resulting in fewer evictions, and more cache hits.

    This notebook will not discuss either how to pick optimal tile sizes for a given kernel, or how to prove that the transformation is correct, and instead explore a possible API and implementation for this transformation in the style of Exo.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.parser import Parser
    from xdsl.context import Context
    from xdsl.dialects import arith, func, test, scf, builtin, memref
    from xdsl.transforms.scf_for_loop_range_folding import ScfForLoopRangeFoldingPass, ScfForLoopRangeFolding
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.pattern_rewriter import PatternRewriter
    import inspect

    return (
        Context,
        Parser,
        arith,
        builtin,
        func,
        inspect,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's the implementation of the `before` function using the `func`, `arith`, `scf`, and `memref` dialects:
    """)
    return


@app.cell(hide_code=True)
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
        r = Rewriter

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

        # inner body stays the same
        inner_body = r.move_region_contents_to_new_regions(cursor.body)
        ii, *iter_args_i = inner_body.block.args

        # outer body contains inner upper bound and inner loop
        outer_body = Region(Block(arg_types=(ii.type, *(val.type for val in iter_args_i))))
        io, *iter_args_o = outer_body.block.args

        # inner upper bound is `step` more than i_o
        inner_ub = arith.AddiOp(io, step_op)
        r.insert_op(inner_ub, InsertPoint.at_start(outer_body.block))

        # i_o increments by original step
        inner_loop = scf.ForOp(io, inner_ub, cursor.step, iter_args_o, inner_body)
        r.insert_op(inner_loop, InsertPoint.after(inner_ub))

        # outer loop increments by new step
        outer_loop = scf.ForOp(cursor.lb, cursor.ub, step_op, cursor.iter_args, outer_body)

        # replace op
        r.replace_op(cursor, outer_loop)

        for old, new_o_res in zip(res_names, outer_loop.results):
            if old is not None:
                new_o_res.name_hint = old + "_o"

        for (outer_arg, inner_arg) in zip(outer_body.block.args, inner_body.block.args, strict=True):
            if inner_arg.name_hint is not None:
                outer_arg.name_hint = inner_arg.name_hint + "_o"
                inner_arg.name_hint += "_i"

        inner_ub.result.name_hint = "inner_ub" if io.name_hint is None else f"{io.name_hint}_ub"

        return outer_loop, inner_loop

    return (split,)


@app.cell
def _(inspect, split):
    lines = inspect.getsource(split)
    print(lines)
    return


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

        old_i_args = i_loop.body.block.args
        old_o_args = o_loop.body.block.args
        old_i_ind_var = old_i_args[0]
        old_o_ind_var = old_o_args[0]

        new_i_body = r.move_region_contents_to_new_regions(i_loop.body)
        new_i_loop = scf.ForOp(o_loop.lb, o_loop.ub, o_loop.step, (), new_i_body)
        new_o_body  = Region(Block([new_i_loop], arg_types=(o_loop.body.block.args[0].type,)))
        new_o_loop = scf.ForOp(i_loop.lb, i_loop.ub, i_loop.step, (), new_o_body)
        r.replace_op(o_loop, new_o_loop)

        new_i_args = new_i_body.block.args
        new_o_args = new_o_body.block.args

        for (old_inner, new_outer) in zip(new_i_loop.body.block.args, new_o_loop.body.block.args, strict=True):
            new_outer.name_hint = old_inner.name_hint

        for (old_outer, new_inner) in zip(o_loop.body.block.args, new_i_loop.body.block.args, strict=True):
            new_inner.name_hint = old_outer.name_hint

        # We swapped the bodies and must now swap the induction arguments
        # The new inner induction argument is the one that used to be inner, which has uses
        # The new outer just got created, and does not have uses
        # First replace uses of old inner with new outer
        # Now old inner (which is also the new inner) has now uses
        # Then replace uses of old outer with new inner

        new_i_ind_var = new_i_args[0]
        new_o_ind_var = new_o_args[0]

        assert new_i_ind_var is old_i_ind_var

        old_i_ind_var.replace_all_uses_with(new_o_ind_var)
        old_o_ind_var.replace_all_uses_with(new_i_ind_var)

        return new_o_loop, new_i_loop

    return (reorder_loops,)


@app.cell
def _(ModuleOp, reorder_loops, scf, split):
    def tile_2d(module: ModuleOp, i: scf.ForOp, j: scf.ForOp) -> tuple[scf.ForOp, scf.ForOp, scf.ForOp, scf.ForOp]:
        io, ii = split(module, i, 16)
        jo, ji = split(module, j, 8)
        reorder_loops(module, ii, jo)
        return io, jo, ii, ji

    return (tile_2d,)


@app.cell
def _(Printer, find, input_module, scf, tile_2d):
    # Tile 2D rewrite

    print("IR before tiling:")
    Printer().print_op(input_module)
    _module =input_module.clone()

    i, j, k = find(_module, scf.ForOp)
    io, ii, jo, ji = tile_2d(_module, i, j)

    print("\n")
    print("IR after tiling:")
    Printer().print_op(_module)
    return


if __name__ == "__main__":
    app.run()
