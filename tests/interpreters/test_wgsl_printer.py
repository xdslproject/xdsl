from io import StringIO
from xdsl.dialects import arith, memref
from xdsl.dialects.builtin import IndexType, IntegerAttr, i32
from xdsl.interpreters.experimental.wgsl_printer import WGSLPrinter
from xdsl.utils.test_value import TestSSAValue


def test_arith_constant():
    file = StringIO("")

    cst = arith.Constant(IntegerAttr(42, IndexType()))

    printer = WGSLPrinter()
    printer.print(cst, file)

    assert "let v0 : u32 = 42u;" in file.getvalue()


def test_memref_load():
    file = StringIO("")

    memref_type = memref.MemRefType.from_element_type_and_shape(i32, [10, 10])

    memref_val = TestSSAValue(memref_type)
    x = arith.Constant(IntegerAttr(2, IndexType()))
    y = arith.Constant(IntegerAttr(4, IndexType()))

    load = memref.Load.get(memref_val, [x, y])

    printer = WGSLPrinter()
    printer.print(load, file)

    assert "let v1 = v0[v2, v3];" in file.getvalue()
