import inspect
from snake_egg import EGraph, Rewrite, Var, vars
from snake_egg import *

from collections import namedtuple

from xdsl.ir import *
from xdsl.parser import *
from xdsl.printer import *
from xdsl.dialects.std import *
from xdsl.dialects.scf import *
from xdsl.dialects.arith import *
from xdsl.dialects.affine import *
from xdsl.dialects.builtin import *

EgraphValueMap = dict

# define nodes for egraph structure
arith_constant = namedtuple('arith_constant', 'c') 
arith_andi = namedtuple('arith_andi', 'a b') 
arith_ori = namedtuple('arith_ori', 'a b') 
arith_xori = namedtuple('arith_xori', 'a b') 
foo = namedtuple('foo', 'x y z')

a, b, _ = vars('a b _')
rules = [
    Rewrite(arith_andi(a, b),                                       arith_andi(b, a)),
    Rewrite(arith_andi(arith_constant(0), arith_constant(0)),       arith_constant(0)),
    Rewrite(arith_andi(_, arith_constant(0)),                       arith_constant(0)),
    Rewrite(arith_andi(arith_constant(0), _),                       arith_constant(0)),

    Rewrite(arith_ori(a, b),                                        arith_ori(b, a)),
    Rewrite(arith_ori(_, arith_constant(1)),                        arith_constant(1)),
    Rewrite(arith_ori(arith_constant(1), _),                        arith_constant(1)),

    Rewrite(arith_xori(a, b),                                       arith_xori(b, a)),
    Rewrite(arith_xori(arith_constant(1), arith_constant(1)),       arith_constant(0)),
    Rewrite(arith_xori(arith_constant(0), arith_constant(0)),       arith_constant(0)),
    Rewrite(arith_xori(arith_constant(1), arith_constant(0)),       arith_constant(1)),
    Rewrite(arith_xori(arith_constant(0), arith_constant(1)),       arith_constant(1)),
]


def parse(prog: str) -> ModuleOp():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)
    scf = Scf(ctx)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    return module


def add_to_egraph(op: Operation, nodeContext: EgraphValueMap, egraph: EGraph):
    match op:
        case ModuleOp():
            mod: ModuleOp = op
            print("module:")
            for operation in mod.ops:
                add_to_egraph(operation, nodeContext, egraph)
            return
        case FuncOp():
            func: FuncOp = op
            print("func:")
            for block in func.body.blocks:
                for operation in block.ops:
                    add_to_egraph(operation, nodeContext, egraph)
            return
        case Call():
            print("call")
            return
        case Constant():
            const : Constant = op
            constNode = egraph.add(arith_constant(const.value.value))
            nodeContext[const] = constNode
            print(str(constNode) + " - constant: " +  str(const.value.value))
            return
        case AndI():
            andi : AndI = op
            andiNode = egraph.add(arith_andi(nodeContext[andi.input1.op], nodeContext[andi.input2.op]))
            nodeContext[andi] = andiNode
            print(str(andiNode) + " - andi(" + str(nodeContext[andi.input1.op]) + "," + str(nodeContext[andi.input2.op]) + ")")
            return
        case OrI():
            ori : OrI = op
            oriNode = egraph.add(arith_ori(nodeContext[ori.input1.op], nodeContext[ori.input2.op]))
            nodeContext[ori] = oriNode
            print(str(oriNode) + " - ori(" + str(nodeContext[ori.input1.op]) + "," + str(nodeContext[ori.input2.op]) + ")")
            return
        case XOrI():
            xori : XOrI = op
            xoriNode = egraph.add(arith_xori(nodeContext[xori.input1.op], nodeContext[xori.input2.op]))
            nodeContext[xori] = xoriNode
            print(str(xoriNode) + " - xori(" + str(nodeContext[xori.input1.op]) + "," + str(nodeContext[xori.input2.op]) + ")")
            return
        case _:
            return


prog = """
module() {
  builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"]
  {

    %7 : !i1 = arith.constant() ["value" = 0 : !i1]
    %8 : !i1 = arith.constant() ["value" = 1 : !i1]
    %9 : !i1 = arith.andi(%7 : !i1, %8 : !i1)
    %10 : !i1 = arith.ori(%7 : !i1, %8 : !i1)
    %11 : !i1 = arith.xori(%7 : !i1, %8 : !i1)
  }

  builtin.func() ["sym_name" = "rec", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"]
  {
  ^1(%20: !i32):
    %21 : !i32 = std.call(%20 : !i32) ["callee" = @rec] 
    std.return(%21 :!i32)
  }
}
"""

module = parse(prog)

printer = Printer()
printer.print_op(module)

egraph = EGraph()
add_to_egraph(module, {}, egraph)

egraph.run(rules, iter_limit=1)
# TODO print the egraph