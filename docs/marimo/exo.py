import marimo

__generated_with = "0.12.6"
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
    func.func @hello() {
        %c0 = arith.constant 0 : index
        %c100 = arith.constant 100 : index
        %c1 = arith.constant 1 : index
        %c200 = arith.constant 200 : index

        scf.for %i = %c0 to %c100 step %c1 {
          scf.for %j = %c0 to %c200 step %c1 {
              "test.op"(%i, %j) : (index, index) -> ()
          }
        }
        func.return
    }
    """

    input_module = Parser(ctx, input_str).parse_module()
    xmo.module_html(input_module)
    return input_module, input_str


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

        step_op = arith.ConstantOp(builtin.IntegerAttr(div_const, cursor.step.type))
        r.insert_op(step_op, insertion_point=InsertPoint.before(cursor))

        inner_body = r.move_region_contents_to_new_regions(cursor.body) # this is region
        ii = inner_body.block.args[0]
        ii.name_hint = "ii"
        inner_loop = scf.ForOp(cursor.lb, step_op.result, cursor.step, (), inner_body)

        outer_body = Region(Block([inner_loop], arg_types=(ii.type,)))
        io = outer_body.block.args[0]
        io.name_hint = "io"

        outer_loop = scf.ForOp(cursor.lb, cursor.ub, step_op, [], outer_body)
        new_i = arith.AddiOp(inner_body.block.args[0], outer_body.block.args[0])
        r.insert_op(new_i, insertion_point=InsertPoint.at_start(inner_body.block))
        new_i.result.name_hint = "i"

        ii.replace_by_if(new_i.result, lambda val : val.operation != new_i)

        r.replace_op(cursor, outer_loop, new_results=())


    print("IR before split")
    Printer().print_op(input_module)
    _module =input_module.clone()

    cursors = find(_module, "scf.for") # this should be loops
    split(_module, cursors[0], 10) # split the loop cursors[0] is pointing to by 8
    split(_module, cursors[1], 20)

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
