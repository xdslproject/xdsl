from io import StringIO

from xdsl.backend.wgsl.wgsl_printer import WGSLPrinter
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, gpu, memref, test
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, i32
from xdsl.parser import Parser
from xdsl.utils.test_value import TestSSAValue

lhs_op = test.TestOp(result_types=[IndexType()])
rhs_op = test.TestOp(result_types=[IndexType()])


def test_gpu_global_id():
    file = StringIO("")

    global_id_x = gpu.GlobalIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))

    printer = WGSLPrinter()
    printer.print(global_id_x, file)

    assert "let v0: u32 = global_invocation_id.x;" in file.getvalue()


def test_gpu_thread_id():
    file = StringIO("")

    thread_id_x = gpu.ThreadIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))

    printer = WGSLPrinter()
    printer.print(thread_id_x, file)

    assert "let v0: u32 = local_invocation_id.x;" in file.getvalue()


def test_gpu_block_id():
    file = StringIO("")

    block_id_x = gpu.BlockIdOp(gpu.DimensionAttr(gpu.DimensionEnum.X))

    printer = WGSLPrinter()
    printer.print(block_id_x, file)

    assert "let v0: u32 = workgroup_id.x;" in file.getvalue()


def test_gpu_grid_dim():
    file = StringIO("")

    num_workgroups = gpu.GridDimOp(gpu.DimensionAttr(gpu.DimensionEnum.X))

    printer = WGSLPrinter()
    printer.print(num_workgroups, file)

    assert "let v0: u32 = num_workgroups.x;" in file.getvalue()


def test_arith_constant_unsigned():
    file = StringIO("")

    cst = arith.ConstantOp(IntegerAttr(42, IndexType()))

    printer = WGSLPrinter()
    printer.print(cst, file)

    assert "let v0 : u32 = 42u;" in file.getvalue()


def test_arith_constant_unsigned_neg():
    file = StringIO("")

    cst = arith.ConstantOp(IntegerAttr(-1, IndexType()))
    cst.result.name_hint = "temp"

    printer = WGSLPrinter()
    printer.print(cst, file)

    assert "let vtemp : u32 = 4294967295u;" in file.getvalue()


def test_arith_constant_signed():
    file = StringIO("")

    cst = arith.ConstantOp(IntegerAttr(42, IntegerType(32)))
    cst.result.name_hint = "temp"

    printer = WGSLPrinter()
    printer.print(cst, file)

    assert "let vtemp : i32 = 42;" in file.getvalue()


def test_arith_addi():
    file = StringIO("")

    addi = arith.AddiOp(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(addi, file)

    assert "let v0 = v1 + v2;" in file.getvalue()


def test_arith_subi():
    file = StringIO("")

    subi = arith.SubiOp(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(subi, file)

    assert "let v0 = v1 - v2;" in file.getvalue()


def test_arith_muli():
    file = StringIO("")

    muli = arith.MuliOp(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(muli, file)

    assert "let v0 = v1 * v2;" in file.getvalue()


def test_arith_addf():
    file = StringIO("")

    addf = arith.AddfOp(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(addf, file)

    assert "let v0 = v1 + v2;" in file.getvalue()


def test_arith_subf():
    file = StringIO("")

    subf = arith.SubfOp(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(subf, file)

    assert "let v0 = v1 - v2;" in file.getvalue()


def test_arith_mulf():
    file = StringIO("")

    mulf = arith.MulfOp(lhs_op, rhs_op)

    printer = WGSLPrinter()
    printer.print(mulf, file)

    assert "let v0 = v1 * v2;" in file.getvalue()


def test_memref_load():
    file = StringIO("")

    memref_type = memref.MemRefType(i32, [10, 10])

    memref_val = TestSSAValue(memref_type)

    load = memref.LoadOp.get(memref_val, [lhs_op.res[0], rhs_op.res[0]])

    printer = WGSLPrinter()
    printer.print(load, file)

    assert "let v1 = v0[10u * v1 + 1u * v2];" in file.getvalue()


def test_memref_store():
    file = StringIO("")

    memref_type = memref.MemRefType(i32, [10, 10])

    memref_val = TestSSAValue(memref_type)

    load = memref.LoadOp.get(memref_val, [lhs_op.res[0], rhs_op.res[0]])

    store = memref.StoreOp.get(load.res, memref_val, [lhs_op.res[0], rhs_op.res[0]])

    printer = WGSLPrinter()
    printer.print(store, file)

    assert "v1[10u * v1 + 1u * v2] = v0;" in file.getvalue()


def test_2d5pt():
    mlir_source = """
builtin.module attributes {gpu.container_module} {
  "gpu.module"() ({
    "gpu.func"() ({
    ^0(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<260x260xf32>, %arg4 : index, %arg5 : f32, %arg6 : f32, %arg7 : f32, %arg8 : f32, %arg9 : f32, %arg10 : f32, %arg11 : memref<260x260xf32>, %arg12: memref<260x260xindex>):
      %0 = "arith.constant"() {"value" = 2 : index} : () -> index
      %1 = "gpu.block_id"() {"dimension" = #gpu<dim x>} : () -> index
      %2 = "gpu.block_id"() {"dimension" = #gpu<dim y>} : () -> index
      %3 = "gpu.thread_id"() {"dimension" = #gpu<dim x>} : () -> index
      %4 = "gpu.thread_id"() {"dimension" = #gpu<dim y>} : () -> index
      %5 = arith.muli %1, %arg0 : index
      %6 = arith.addi %5, %arg1 : index
      %7 = arith.muli %2, %arg2 : index
      %8 = arith.addi %7, %arg1 : index
      %9 = arith.muli %3, %arg2 : index
      %10 = arith.addi %9, %arg1 : index
      %11 = arith.muli %4, %arg2 : index
      %12 = arith.addi %11, %arg1 : index
      %13 = arith.addi %10, %6 : index
      %14 = arith.addi %12, %8 : index
      %15 = arith.addi %14, %0 : index
      %16 = arith.addi %13, %0 : index
      %17 = "memref.load"(%arg3, %15, %16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %18 = arith.addi %14, %arg4 : index
      %19 = arith.addi %18, %0 : index
      %20 = "memref.load"(%arg3, %19, %16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %21 = arith.addi %14, %arg2 : index
      %22 = arith.addi %21, %0 : index
      %23 = "memref.load"(%arg3, %22, %16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %24 = arith.addi %13, %arg4 : index
      %25 = arith.addi %24, %0 : index
      %26 = "memref.load"(%arg3, %15, %25) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %27 = arith.addi %13, %arg2 : index
      %28 = arith.addi %27, %0 : index
      %29 = "memref.load"(%arg3, %15, %28) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %30 = arith.mulf %17, %arg5 : f32
      %31 = arith.mulf %20, %arg6 : f32
      %32 = arith.mulf %23, %arg6 : f32
      %33 = arith.mulf %17, %arg7 : f32
      %34 = arith.addf %31, %32 : f32
      %35 = arith.addf %34, %33 : f32
      %36 = arith.mulf %26, %arg6 : f32
      %37 = arith.mulf %29, %arg6 : f32
      %38 = arith.addf %36, %37 : f32
      %temp = arith.addf %38, %33 : f32
      %40 = arith.addf %35, %temp : f32
      %41 = arith.mulf %40, %arg8 : f32
      %42 = arith.addf %30, %arg9 : f32
      %43 = arith.addf %42, %41 : f32
      %44 = arith.mulf %43, %arg10 : f32
      "memref.store"(%44, %arg11, %15, %16) {"nontemporal" = false} : (f32, memref<260x260xf32>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (index, index, index, memref<260x260xf32>, index, f32, f32, f32, f32, f32, f32, memref<260x260xf32>, memref<260x260xindex>) -> (),
        "gpu.kernel", "gpu.known_block_size" = array<i32: 128, 1, 1>, "gpu.known_grid_size" = array<i32: 2, 256, 1>,
        "sym_name" = "apply_kernel_kernel",
        "workgroup_attributions" = 0 : i64
       } : () -> ()
    "gpu.module_end"() : () -> ()
  }) {"sym_name" = "apply_kernel_kernel"} : () -> ()
}
"""

    expected = """
    @group(0) @binding(0)
    var<storage,read> varg0: u32;

    @group(0) @binding(1)
    var<storage,read> varg1: u32;

    @group(0) @binding(2)
    var<storage,read> varg2: u32;

    @group(0) @binding(3)
    var<storage,read> varg3: array<f32>;

    @group(0) @binding(4)
    var<storage,read> varg4: u32;

    @group(0) @binding(5)
    var<storage,read> varg5: f32;

    @group(0) @binding(6)
    var<storage,read> varg6: f32;

    @group(0) @binding(7)
    var<storage,read> varg7: f32;

    @group(0) @binding(8)
    var<storage,read> varg8: f32;

    @group(0) @binding(9)
    var<storage,read> varg9: f32;

    @group(0) @binding(10)
    var<storage,read> varg10: f32;

    @group(0) @binding(11)
    var<storage,read_write> varg11: array<f32>;

    @group(0) @binding(12)
    var<storage,read> varg12: array<u32>;

    @compute
    @workgroup_size(128,1,1)
    fn apply_kernel_kernel(@builtin(global_invocation_id) global_invocation_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>) {

        let v0 : u32 = 2u;
        let v1: u32 = workgroup_id.x;
        let v2: u32 = workgroup_id.y;
        let v3: u32 = local_invocation_id.x;
        let v4: u32 = local_invocation_id.y;
        let v5 = v1 * varg0;
        let v6 = v5 + varg1;
        let v7 = v2 * varg2;
        let v8 = v7 + varg1;
        let v9 = v3 * varg2;
        let v10 = v9 + varg1;
        let v11 = v4 * varg2;
        let v12 = v11 + varg1;
        let v13 = v10 + v6;
        let v14 = v12 + v8;
        let v15 = v14 + v0;
        let v16 = v13 + v0;
        let v17 = varg3[260u * v15 + 1u * v16];
        let v18 = v14 + varg4;
        let v19 = v18 + v0;
        let v20 = varg3[260u * v19 + 1u * v16];
        let v21 = v14 + varg2;
        let v22 = v21 + v0;
        let v23 = varg3[260u * v22 + 1u * v16];
        let v24 = v13 + varg4;
        let v25 = v24 + v0;
        let v26 = varg3[260u * v15 + 1u * v25];
        let v27 = v13 + varg2;
        let v28 = v27 + v0;
        let v29 = varg3[260u * v15 + 1u * v28];
        let v30 = v17 * varg5;
        let v31 = v20 * varg6;
        let v32 = v23 * varg6;
        let v33 = v17 * varg7;
        let v34 = v31 + v32;
        let v35 = v34 + v33;
        let v36 = v26 * varg6;
        let v37 = v29 * varg6;
        let v38 = v36 + v37;
        let vtemp = v38 + v33;
        let v39 = v35 + vtemp;
        let v40 = v39 * varg8;
        let v41 = v30 + varg9;
        let v42 = v41 + v40;
        let v43 = v42 * varg10;
        varg11[260u * v15 + 1u * v16] = v43;
            }
"""
    context = MLContext()
    context.load_dialect(arith.Arith)
    context.load_dialect(memref.MemRef)
    context.load_dialect(builtin.Builtin)
    context.load_dialect(gpu.GPU)
    parser = Parser(context, mlir_source)
    module = parser.parse_module()

    gpu_module = module.ops.first

    file = StringIO("")
    printer = WGSLPrinter()
    printer.print(gpu_module, file)

    assert expected in file.getvalue()
