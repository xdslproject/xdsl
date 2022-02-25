import inspect
from snake_egg import EGraph, Rewrite, Var, vars
from snake_egg import *

from collections import namedtuple
from typing import NamedTuple

from xdsl.ir import *
from xdsl.parser import *
from xdsl.printer import *
from xdsl.dialects.std import *
from xdsl.dialects.scf import *
from xdsl.dialects.arith import *
from xdsl.dialects.affine import *
from xdsl.dialects.builtin import *



def namedENode(typename, field_names, *, rename=False, defaults=None, module=None):
    def __repr__custom(self):
        return 'custom repr'
    def __str__custom(self):
        return 'custom str'

    result = namedtuple(typename, field_names, rename=False, defaults=None, module=None)
    result.__repr__ = __repr__custom
    result.__str__ = __str__custom
    # setattr(result, "__repr__", __repr__custom)
    # setattr(result, "__str__", __str__custom)

    return result


EgraphValueMap = dict
eNodes = {}

# define nodes for egraph structure
arith_constant = namedENode('arith_constant', 'c')
arith_andi = namedENode('arith_andi', 'a b')
arith_ori = namedENode('arith_ori', 'a b')
arith_xori = namedENode('arith_xori', 'a b')
foo = namedENode('foo', 'x y z')

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

def add_to_egraph(ctx: MLContext, op: Operation, nodeContext: EgraphValueMap, egraph: EGraph):
    for registered_op in ctx._registeredOps:
        opName = registered_op.replace(".", "_")
        print(opName)
        # if operation has operands
        operands = []
        if ctx._registeredOps[registered_op].irdl_operand_defs:
            operands = list(map(str, list(zip(*ctx._registeredOps[registered_op].irdl_operand_defs))[0]))
            print("operands:")
            print(operands)
        if not ctx._registeredOps[registered_op].irdl_region_defs:
            eNodes[opName] = namedENode(opName, operands)

    add_to_egraph_rec(ctx, op, nodeContext, egraph)
    for enode in eNodes:
        # not sure why these printing forms are not identical. 
        print(eNodes[enode].__str__(eNodes[enode]))
        print(eNodes[enode])

def add_to_egraph_rec(ctx: MLContext, op: Operation, nodeContext: EgraphValueMap, egraph: EGraph):
    match op:
        case ModuleOp():
            mod: ModuleOp = op
            print("module:")
            for operation in mod.ops:
                add_to_egraph_rec(ctx, operation, nodeContext, egraph)
            return
        case FuncOp():
            func: FuncOp = op
            print("func:")
            for block in func.body.blocks:
                for operation in block.ops:
                    add_to_egraph_rec(ctx, operation, nodeContext, egraph)
            return
        case Call():
            print("call")
            return
        case Constant():
            const : Constant = op
            constNode = egraph.add(arith_constant(const.value.value.data))
            nodeContext[const] = constNode
            print(str(constNode) + " - constant: " +  str(const.value.value.data))
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


            