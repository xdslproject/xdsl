import xdsl.dialects.linalg as Linalg
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import Region, f32, Block, AffineMapAttr
import xdsl.dialects.func as Func
from xdsl.ir.affine import AffineMap, AffineExpr
import xdsl.dialects.memref as MemRef
from xdsl.printer import Printer


def test_linalg_on_memrefs():
    memref = MemRef.Alloc.get(f32, shape=[100])
    constant = Constant.from_float_and_width(0.0, f32)

    inputs = [constant.results[0]]
    outputs = [memref.results[0]]

    bodyblock = Block(arg_types=[f32, f32])
    bodyblock.add_op(Linalg.Yield.get(bodyblock.args[0]))
    body = Region(bodyblock)

    indexing = AffineExpr.dimension(0)
    indexing_map = AffineMap(1, 0, [indexing])

    indexing_maps = [AffineMapAttr(AffineMap(1, 0, [])), AffineMapAttr(indexing_map)]

    iterators = [Linalg.IteratorTypeAttr(Linalg.IteratorType.PARALLEL)]

    op = Linalg.Generic.get(inputs, outputs, body, indexing_maps, iterators)

    # Create a block with all these operations inside it.
    block = Block()
    block.add_op(memref)
    block.add_op(constant)
    block.add_op(op)
    block.add_op(Func.Return())
    region = Region(block)

    func = Func.FuncOp("foo", ([], []), region)

    printer = Printer()
    printer.print(func)
