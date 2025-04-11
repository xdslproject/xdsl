import marimo

__generated_with = "0.12.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.parser import Parser
    from xdsl.context import Context
    from xdsl.dialects import arith, func, test, scf, builtin, memref
    return Context, Parser, arith, builtin, func, memref, mo, scf, test, xmo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exo-Style Scheduling in xDSL

        It works.
        """
    )
    return


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
    return input_module, input_str


@app.cell(hide_code=True)
def _(Parser, ctx, xmo):
    input_one_str = """
    func.func @hello() -> index {
        %c0 = arith.constant 0 : index
        %c100 = arith.constant 100 : index
        %c1 = arith.constant 1 : index
        %c200 = arith.constant 200 : index
        %acc_init = arith.constant 0 : index

        %res = scf.for %i = %c0 to %c100 step %c1 iter_args(%acc_in = %acc_init) -> (index) {
            %acc_out = arith.addi %acc_in, %i : index
            scf.yield %acc_out : index
        }
        func.return %res : index
    }
    """

    input_one_module = Parser(ctx, input_one_str).parse_module()
    input_one_module.verify()
    xmo.module_html(input_one_module)
    return input_one_module, input_one_str


@app.cell(hide_code=True)
def _(Parser, ctx, xmo):
    output_one_str = """
    func.func @hello() -> index {
        %c0 = arith.constant 0 : index
        %c100 = arith.constant 100 : index
        %c1 = arith.constant 1 : index
        %c200 = arith.constant 200 : index
        %acc_init = arith.constant 0 : index
        %c5 = arith.constant 5 : index

        %res_o = scf.for %io = %c0 to %c100 step %c5 iter_args(%acc_in_o = %acc_init) -> (index) {
          %acc_out_o = scf.for %ii = %c0 to %c5 step %c1 iter_args(%acc_in_i = %acc_in_o) -> (index) {
              %i = arith.addi %io, %ii : index
              %acc_out_i = arith.addi %acc_in_i, %i : index
              scf.yield %acc_out_i : index
          }
          scf.yield %acc_out_o : index
        }
        func.return %res_o : index

    }
    """

    output_one_module = Parser(ctx, output_one_str).parse_module()
    output_one_module.verify()
    xmo.module_html(output_one_module)
    return output_one_module, output_one_str


@app.cell
def _(input_module):
    from xdsl.dialects.builtin import ModuleOp

    # TODO: needs more sophisticated pattern language not just string match
    def find(module: ModuleOp, pattern: str):
        return list(op for op in module.walk() if op.name == pattern)

    find(input_module, "scf.for")
    return ModuleOp, find


@app.cell
def _(ModuleOp, arith, builtin, find, input_module, scf):
    from typing import cast

    from xdsl.printer import Printer
    from xdsl.ir import OpResult, Operation, Region, Block
    from xdsl.builder import InsertPoint
    from xdsl.pattern_rewriter import (
        Rewriter,
        op_type_rewrite_pattern,
    )
    from dataclasses import dataclass


    def split(module : ModuleOp, cursor: Operation, div_const: int): # cursor is just an Operation for now. We need something better when we support forwarding
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

        res_names = tuple(r.name_hint for r in cursor.results)

        step_op = arith.ConstantOp(builtin.IntegerAttr(div_const, cursor.step.type))
        r.insert_op(step_op, insertion_point=InsertPoint.before(cursor))
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

        print(outer_body.block)
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


    # TODO: reorder_loops does not check the commutativity of the body of the K loop. It needs to be asserted from the user.
    def reorder_loops(module : ModuleOp, cursor : Operation):
        r = Rewriter()

        # cursor must be a loop
        assert isinstance(cursor, scf.ForOp)
        # body of the outer loop should be size 1. Loops must be perfectly nested
        assert len(cursor.body.block.ops) == 2
        assert isinstance(cursor.body.block.first_op, scf.ForOp)

        o_loop = cursor
        i_loop = cursor.body.block.first_op

        # check that inner loop bounds does not depend on the outer loop iteration variable
        # actually we can just check if they're constant or not
        assert isinstance(i_loop.ub.owner, arith.ConstantOp)
        assert isinstance(i_loop.lb.owner, arith.ConstantOp)

        new_body = r.move_region_contents_to_new_regions(i_loop.body)
        new_i_loop = scf.ForOp(o_loop.lb, o_loop.ub, o_loop.step, (), new_body)
        outer_body  = Region(Block([new_i_loop], arg_types=(cursor.body.block.args[0].type,)))
        new_o_loop = scf.ForOp(i_loop.lb, i_loop.ub, i_loop.step, (), outer_body)
        r.replace_op(cursor, new_o_loop)


    print("IR before split")
    Printer().print_op(input_module)
    _module =input_module.clone()

    cursors = find(_module, "scf.for") # this should be loops
    split(_module, cursors[0], 16) # split the loop cursors[0] is pointing to by 8
    # split(_module, cursors[1], 8)
    #reorder_loops(_module, cursors[0])

    print("\n")
    print("IR after split")
    Printer().print_op(_module)

    # TODO: think about moving the statement outside and inside the loop
    # think about the safety condition of the reordering the loop

    # safety checks for move and reorder_loops
    # 1. check all the statments inside the loop are pure
    # 2. call the verifier to check that there's no use of register before its definition
    return (
        Block,
        InsertPoint,
        OpResult,
        Operation,
        Printer,
        Region,
        Rewriter,
        cast,
        cursors,
        dataclass,
        op_type_rewrite_pattern,
        reorder_loops,
        split,
    )


@app.cell
def _(Parser, ctx, xmo):
    output_str = """
    func.func @hello_2() {
        %c0 = arith.constant 0 : index
        %c100 = arith.constant 100 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c1 = arith.constant 1 : index
        %c200 = arith.constant 200 : index

        scf.for %io = %c0 to %c100 step %c4 {
          scf.for %jo = %c0 to %c200 step %c5 {
              scf.for %il = %c0 to %c4 step %c1 {
                  scf.for %jl = %c0 to %c5 step %c1 {
                      %i2 = arith.addi %io, %il : index
                      %j2 = arith.addi %jo, %jl : index
                      "test.op"(%i2, %j2) : (index, index) -> ()
                  }
              }
          }
        }
        func.return
    }
    """

    output_module = Parser(ctx, output_str).parse_module()
    xmo.module_html(output_module)
    return output_module, output_str


if __name__ == "__main__":
    app.run()
