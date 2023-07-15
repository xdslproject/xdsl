from io import StringIO

from xdsl.dialects import arith, gpu, memref, test
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, i32
from xdsl.interpreters.experimental.wgsl_printer import WGSLPrinter
from xdsl.utils.test_value import TestSSAValue

lhs_op = test.TestOp(operands=[[]], result_types=[IndexType()], regions=[[]])
rhs_op = test.TestOp(operands=[[]], result_types=[IndexType()], regions=[[]])


def test_gpu_global_id():
    file = StringIO("")

    global_id_x = gpu.GlobalIdOp(gpu.DimensionAttr.from_dimension("x"))

    printer = WGSLPrinter()
    printer.print(global_id_x, file)

    assert "let v0: u32 = global_invocation_id.x;" in file.getvalue()


def test_gpu_thread_id():
    file = StringIO("")

    thread_id_x = gpu.ThreadIdOp(gpu.DimensionAttr.from_dimension("x"))

    printer = WGSLPrinter()
    printer.print(thread_id_x, file)

    assert "let v0: u32 = local_invocation_id.x;" in file.getvalue()


def test_gpu_block_id():
    file = StringIO("")

    block_id_x = gpu.BlockIdOp(gpu.DimensionAttr.from_dimension("x"))

    printer = WGSLPrinter()
    printer.print(block_id_x, file)

    assert "let v0: u32 = workgroup_id.x;" in file.getvalue()


def test_gpu_grid_dim():
    file = StringIO("")

    num_workgroups = gpu.GridDimOp(gpu.DimensionAttr.from_dimension("x"))

    printer = WGSLPrinter()
    printer.print(num_workgroups, file)

    assert "let v0: u32 = num_workgroups.x;" in file.getvalue()


def test_arith_constant_unsigned():
    file = StringIO("")

    cst = arith.Constant(IntegerAttr(42, IndexType()))

    printer = WGSLPrinter()
    printer.print(cst, file)

    assert "let v0 : u32 = 42u;" in file.getvalue()


def test_arith_constant_unsigned_neg():
    file = StringIO("")

    cst = arith.Constant(IntegerAttr(-1, IndexType()))
    cst.result.name_hint = "temp"

    printer = WGSLPrinter()
    printer.print(cst, file)

    assert "let vtemp : u32 = 4294967295u;" in file.getvalue()


def test_arith_constant_signed():
    file = StringIO("")

    cst = arith.Constant(IntegerAttr(42, IntegerType(32)))
    cst.result.name_hint = "temp"

    printer = WGSLPrinter()
    printer.print(cst, file)

    assert "let vtemp : i32 = 42;" in file.getvalue()


def test_arith_addi():
    file = StringIO("")

    addi = arith.Addi(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(addi, file)

    assert "let v0 = v1 + v2;" in file.getvalue()


def test_arith_subi():
    file = StringIO("")

    subi = arith.Subi(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(subi, file)

    assert "let v0 = v1 - v2;" in file.getvalue()


def test_arith_muli():
    file = StringIO("")

    muli = arith.Muli(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(muli, file)

    assert "let v0 = v1 * v2;" in file.getvalue()


def test_arith_addf():
    file = StringIO("")

    addf = arith.Addf(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(addf, file)

    assert "let v0 = v1 + v2;" in file.getvalue()


def test_arith_subf():
    file = StringIO("")

    subf = arith.Subf(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(subf, file)

    assert "let v0 = v1 - v2;" in file.getvalue()


def test_arith_mulf():
    file = StringIO("")

    mulf = arith.Mulf(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(mulf, file)

    assert "let v0 = v1 * v2;" in file.getvalue()


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


def test_memref_store():
    file = StringIO("")

    memref_type = memref.MemRefType.from_element_type_and_shape(i32, [10, 10])

    memref_val = TestSSAValue(memref_type)
    x = arith.Constant(IntegerAttr(2, IndexType()))
    y = arith.Constant(IntegerAttr(4, IndexType()))

    load = memref.Load.get(memref_val, [x, y])

    store = memref.Store.get(load.res, memref_val, [x, y])

    printer = WGSLPrinter()
    printer.print(store, file)

    assert "v1[v2, v3] = v0;" in file.getvalue()
