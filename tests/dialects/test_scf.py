from io import StringIO

from xdsl.ir import MLContext
from xdsl.dialects.arith import Constant, Arith
from xdsl.dialects.builtin import Region, IndexType, ModuleOp, Builtin
from xdsl.dialects.cf import Block
from xdsl.dialects.scf import For, ParallelOp, If, Yield, Scf

from xdsl.printer import Printer
from xdsl.parser import Parser, Source


def test_for():
    lb = Constant.from_int_and_width(0, IndexType())
    ub = Constant.from_int_and_width(42, IndexType())
    step = Constant.from_int_and_width(3, IndexType())
    carried = Constant.from_int_and_width(1, IndexType())
    bodyblock = Block(arg_types=[IndexType()])
    body = Region(bodyblock)
    f = For.get(lb, ub, step, [carried], body)

    assert f.lb is lb.result
    assert f.ub is ub.result
    assert f.step is step.result
    assert f.iter_args == tuple([carried.result])
    assert f.body is body

    assert len(f.results) == 1
    assert f.results[0].typ == carried.result.typ
    assert f.operands == (lb.result, ub.result, step.result, carried.result)
    assert f.regions == [body]
    assert f.attributes == {}


def test_parallel():
    lbi = Constant.from_int_and_width(0, IndexType())
    lbj = Constant.from_int_and_width(1, IndexType())
    lbk = Constant.from_int_and_width(18, IndexType())

    ubi = Constant.from_int_and_width(10, IndexType())
    ubj = Constant.from_int_and_width(110, IndexType())
    ubk = Constant.from_int_and_width(92, IndexType())

    si = Constant.from_int_and_width(1, IndexType())
    sj = Constant.from_int_and_width(3, IndexType())
    sk = Constant.from_int_and_width(8, IndexType())

    body = Region()

    lowerBounds = [lbi, lbj, lbk]
    upperBounds = [ubi, ubj, ubk]
    steps = [si, sj, sk]

    p = ParallelOp.get(lowerBounds, upperBounds, steps, body)

    assert isinstance(p, ParallelOp)
    assert p.lowerBound == tuple(l.result for l in lowerBounds)
    assert p.upperBound == tuple(l.result for l in upperBounds)
    assert p.step == tuple(l.result for l in steps)
    assert p.body is body


def test_empty_else():
    m = ModuleOp.from_region_or_ops([
        t := Constant.from_int_and_width(1, 1),
        If.get(t, [], [
            Yield.get(),
        ]),
    ])
    io = StringIO()
    p = Printer(target=Printer.Target.MLIR, stream=io)
    p.print(m)
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Scf)
    p = Parser(ctx, io.getvalue(), Source.MLIR)
    new_module = p.parse_module()

    assert m.is_structurally_equivalent(new_module)
