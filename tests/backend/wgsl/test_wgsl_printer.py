from io import StringIO

from xdsl.backend.wgsl.wgsl_printer import WGSLPrinter
from xdsl.dialects import arith, gpu, memref, test
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, i32
from xdsl.utils.test_value import create_ssa_value

lhs_op = test.TestOp(result_types=[IndexType()])
rhs_op = test.TestOp(result_types=[IndexType()])


def test_gpu_global_id():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    global_id_x = gpu.GlobalIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))
    printer.print(global_id_x)

    assert "let v0: u32 = global_invocation_id.x;" in file.getvalue()


def test_gpu_thread_id():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    thread_id_x = gpu.ThreadIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))
    printer.print(thread_id_x)

    assert "let v0: u32 = local_invocation_id.x;" in file.getvalue()


def test_gpu_block_id():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    block_id_x = gpu.BlockIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))
    printer.print(block_id_x)

    assert "let v0: u32 = workgroup_id.x;" in file.getvalue()


def test_gpu_grid_dim():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    num_workgroups = gpu.GridDimOp(gpu.DimensionAttr(gpu.DimensionEnum.X))
    printer.print(num_workgroups)

    assert "let v0: u32 = num_workgroups.x;" in file.getvalue()


def test_arith_constant_unsigned():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    cst = arith.ConstantOp(IntegerAttr(42, IndexType()))
    printer.print(cst)

    assert "let v0 : u32 = 42u;" in file.getvalue()


def test_arith_constant_unsigned_neg():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    cst = arith.ConstantOp(IntegerAttr(-1, IndexType()))
    cst.result.name_hint = "temp"
    printer.print(cst)

    assert "let vtemp : u32 = 4294967295u;" in file.getvalue()


def test_arith_constant_signed():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    cst = arith.ConstantOp(IntegerAttr(42, IntegerType(32)))
    cst.result.name_hint = "temp"
    printer.print(cst)

    assert "let vtemp : i32 = 42;" in file.getvalue()


def test_arith_addi():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    addi = arith.AddiOp(lhs_op, rhs_op)
    printer.print(addi)

    assert "let v0 = v1 + v2;" in file.getvalue()


def test_arith_subi():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    subi = arith.SubiOp(lhs_op, rhs_op)
    printer.print(subi)

    assert "let v0 = v1 - v2;" in file.getvalue()


def test_arith_muli():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    muli = arith.MuliOp(lhs_op, rhs_op)
    printer.print(muli)

    assert "let v0 = v1 * v2;" in file.getvalue()


def test_arith_addf():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    addf = arith.AddfOp(lhs_op, rhs_op)
    printer.print(addf)

    assert "let v0 = v1 + v2;" in file.getvalue()


def test_arith_subf():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    subf = arith.SubfOp(lhs_op, rhs_op)
    printer.print(subf)

    assert "let v0 = v1 - v2;" in file.getvalue()


def test_arith_mulf():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    mulf = arith.MulfOp(lhs_op, rhs_op)
    printer.print(mulf)

    assert "let v0 = v1 * v2;" in file.getvalue()


def test_memref_load():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    memref_type = memref.MemRefType(i32, [10, 10])
    memref_val = create_ssa_value(memref_type)
    load = memref.LoadOp.get(memref_val, [lhs_op.res[0], rhs_op.res[0]])
    printer.print(load)

    assert "let v1 = v0[10u * v1 + 1u * v2];" in file.getvalue()


def test_memref_store():
    file = StringIO("")
    printer = WGSLPrinter(stream=file)

    memref_type = memref.MemRefType(i32, [10, 10])
    memref_val = create_ssa_value(memref_type)
    load = memref.LoadOp.get(memref_val, [lhs_op.res[0], rhs_op.res[0]])
    store = memref.StoreOp.get(load.res, memref_val, [lhs_op.res[0], rhs_op.res[0]])
    printer.print(store)

    assert "v1[10u * v1 + 1u * v2] = v0;" in file.getvalue()
