import inspect
from snake_egg import EGraph, Rewrite, Var, vars
from snake_egg import *

from collections import namedtuple

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

for r in rules:
    print(r.name)

egraph = EGraph()

# add program: foo(and(0,1), or(0,1), xor(0,1))

_7 = egraph.add(arith_constant(0))
_8 = egraph.add(arith_constant(1))
_9 = egraph.add(arith_andi(_7, _8))
_10 = egraph.add(arith_ori(_7, _8))
_11 = egraph.add(arith_xori(_7, _8))
_12 = egraph.add(foo(_9, _10, _11))

egraph.run(rules, iter_limit=1)

# program was simplified to foo(0,1,1)
def test_foo_simplified():
    assert egraph.equiv(_12, foo(x=arith_constant(c=0), y=arith_constant(c=1), z=arith_constant(c=1)))