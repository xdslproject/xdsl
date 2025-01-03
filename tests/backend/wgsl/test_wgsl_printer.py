from io import StringIO

from xdsl.backend.wgsl.wgsl_printer import WGSLPrinter
from xdsl.dialects import arith, gpu, memref, test
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, i32
from xdsl.utils.test_value import TestSSAValue

lhs_op = test.TestOp(result_types=[IndexType()])
rhs_op = test.TestOp(result_types=[IndexType()])


def test_gpu_global_id():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    global_id_x = gpu.GlobalIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))
    printer.print(global_id_x)

    assert "let v0: u32 = global_invocation_id.x;" in stream.getvalue()


def test_gpu_thread_id():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    thread_id_x = gpu.ThreadIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))
    printer.print(thread_id_x)

    assert "let v0: u32 = local_invocation_id.x;" in stream.getvalue()


def test_gpu_block_id():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    block_id_x = gpu.BlockIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))
    printer.print(block_id_x)

    assert "let v0: u32 = workgroup_id.x;" in stream.getvalue()


def test_gpu_grid_dim():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    num_workgroups = gpu.GridDimOp(gpu.DimensionAttr(gpu.DimensionEnum.X))
    printer.print(num_workgroups)

    assert "let v0: u32 = num_workgroups.x;" in stream.getvalue()


def test_arith_constant_unsigned():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    cst = arith.ConstantOp(IntegerAttr(42, IndexType()))
    printer.print(cst)


def test_arith_constant_unsigned_neg():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    cst = arith.ConstantOp(IntegerAttr(-1, IndexType()))
    cst.result.name_hint = "temp"
    printer.print(cst)

    assert "let vtemp : u32 = 4294967295u;" in stream.getvalue()


def test_arith_constant_signed():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    cst = arith.ConstantOp(IntegerAttr(42, IntegerType(32)))
    cst.result.name_hint = "temp"
    printer.print(cst)

    assert "let vtemp : i32 = 42;" in stream.getvalue()


def test_arith_addi():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    addi = arith.AddiOp(lhs_op, rhs_op)
    printer.print(addi)

    assert "let v0 = v1 + v2;" in stream.getvalue()


def test_arith_subi():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    subi = arith.SubiOp(lhs_op, rhs_op)
    printer.print(subi)

    assert "let v0 = v1 - v2;" in stream.getvalue()


def test_arith_muli():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    muli = arith.MuliOp(lhs_op, rhs_op)
    printer.print(muli)

    assert "let v0 = v1 * v2;" in stream.getvalue()


def test_arith_addf():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    addf = arith.AddfOp(lhs_op, rhs_op)
    printer.print(addf)

    assert "let v0 = v1 + v2;" in stream.getvalue()


def test_arith_subf():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    subf = arith.SubfOp(lhs_op, rhs_op)
    printer.print(subf)

    assert "let v0 = v1 - v2;" in stream.getvalue()


def test_arith_mulf():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    mulf = arith.MulfOp(lhs_op, rhs_op)
    printer.print(mulf)

    assert "let v0 = v1 * v2;" in stream.getvalue()


def test_memref_load():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    memref_type = memref.MemRefType(i32, [10, 10])
    memref_val = TestSSAValue(memref_type)
    load = memref.LoadOp.get(memref_val, [lhs_op.res[0], rhs_op.res[0]])
    printer.print(load)

    assert "let v1 = v0[10u * v1 + 1u * v2];" in stream.getvalue()


def test_memref_store():
    stream = StringIO()
    printer = WGSLPrinter(stream=stream)

    memref_type = memref.MemRefType(i32, [10, 10])
    memref_val = TestSSAValue(memref_type)
    load = memref.LoadOp.get(memref_val, [lhs_op.res[0], rhs_op.res[0]])
    store = memref.StoreOp.get(load.res, memref_val, [lhs_op.res[0], rhs_op.res[0]])
    printer.print(store)

    assert "v1[10u * v1 + 1u * v2] = v0;" in stream.getvalue()
