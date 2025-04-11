import marimo

__generated_with = "0.10.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.parser import Parser
    from xdsl.context import Context
    from xdsl.dialects import arith, func, test, scf, builtin
    return Context, Parser, arith, builtin, func, mo, scf, test, xmo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exo-Style Scheduling in xDSL""")
    return


@app.cell
def _(Context, arith, builtin, func, scf, test):
    ctx = Context()

    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(test.Test)
    return (ctx,)


@app.cell
def _(Parser, ctx, xmo):
    input_str = """
    func.func @hello() -> index {
        %c0 = arith.constant 0 : index
        %c100 = arith.constant 100 : index
        %c1 = arith.constant 1 : index
        %c200 = arith.constant 200 : index
        %acc_init = arith.constant 0 : index

        %res_o = scf.for %i = %c0 to %c100 step %c1 iter_args(%acc_o = %acc_init) -> (index) {
          %res_i = scf.for %j = %c0 to %c200 step %c1 iter_args(%acc_i = %acc_o) -> (index) {
              %res_new = arith.addi %i, %j : index
              scf.yield %acc_i : index
          }
          scf.yield %res_i : index
        }
        func.return %res_o : index
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
def _(ModuleOp, arith, builtin, find, input_one_module, scf):
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
        new_i.result.name_hint = "i"

        ii.replace_by_if(new_i.result, lambda val : val.operation != new_i)

        r.replace_op(cursor, outer_loop)
        for old, new_o_res in zip(res_names, outer_loop.results):
            if old is not None:
                new_o_res.name_hint = old + "_o"

        for (outer_arg, inner_arg) in zip(outer_body.block.args, inner_body.block.args, strict=True):
            if inner_arg.name_hint is not None:
                outer_arg.name_hint = inner_arg.name_hint + "_o"
                inner_arg.name_hint += "_i"

    print("IR before split")
    Printer().print_op(input_one_module)
    _module =input_one_module.clone()

    cursors = find(_module, "scf.for") # this should be loops
    split(_module, cursors[0], 10) # split the loop cursors[0] is pointing to by 8
    # split(_module, cursors[1], 20)

    # TODO: think about moving the statement outside and inside the loop
    # think about the safety condition of the reordering the loop

    # safety checks for move and reorder_loops
    # 1. check all the statments inside the loop are pure
    # 2. call the verifier to check that there's no use of register before its definition

    print("\n")
    print("IR after split")
    Printer().print_op(_module)
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
