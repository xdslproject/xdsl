from xdsl.dialects import snitch_stream
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.snitch_stream import (
    SnitchStreamFunctions,
    StridePattern,
    indexing_map_from_bounds,
)
from xdsl.ir.affine import AffineExpr, AffineMap


def test_indexing_map_constructor():
    assert indexing_map_from_bounds([]) == AffineMap(1, 0, ())
    assert indexing_map_from_bounds([2]) == AffineMap(
        1, 0, (AffineExpr.dimension(0) % 2,)
    )
    assert indexing_map_from_bounds([2, 3]) == AffineMap(
        1, 0, (AffineExpr.dimension(0).floor_div(3) % 2, AffineExpr.dimension(0) % 3)
    )
    assert indexing_map_from_bounds([2, 3, 4]) == AffineMap(
        1,
        0,
        (
            AffineExpr.dimension(0).floor_div(12) % 2,
            AffineExpr.dimension(0).floor_div(4) % 3,
            AffineExpr.dimension(0) % 4,
        ),
    )


def test_riscv_interpreter():
    # register = riscv.IntRegisterType.unallocated()
    # fregister = riscv.FloatRegisterType.unallocated()

    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(SnitchStreamFunctions())

    stride_pattern_op = snitch_stream.StridePatternOp(
        ArrayAttr((IntAttr(2), IntAttr(3))),
        ArrayAttr((IntAttr(12), IntAttr(4))),
        IntAttr(0),
    )

    assert interpreter.run_op(stride_pattern_op, ()) == (
        StridePattern([2, 3], [12, 4]),
    )

    # Generic
    # Strided read
    # Strided write
