// RUN: xdsl-opt %s -t mlir -p stencil-shape-inference,convert-stencil-to-gpu | filecheck %s


"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
    %3 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %4 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %6 = "stencil.load"(%3) : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    %8 = "stencil.apply"(%6) ({
    ^1(%9 : !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, 1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, -1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %15 = "arith.addf"(%10, %11) : (f64, f64) -> f64
      %16 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %17 = "arith.addf"(%15, %16) : (f64, f64) -> f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %18 = "arith.mulf"(%14, %cst) : (f64, f64) -> f64
      %19 = "arith.addf"(%18, %17) : (f64, f64) -> f64
      "stencil.return"(%19) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    "stencil.store"(%8, %4) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0: i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()


// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>):
// CHECK-NEXT:     %2 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %3 = "memref.cast"(%2) : (memref<72x72x72xf64>) -> memref<*xf64>
// CHECK-NEXT:     "gpu.host_register"(%3) : (memref<*xf64>) -> ()
// CHECK-NEXT:     %4 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %5 = "memref.cast"(%4) : (memref<72x72x72xf64>) -> memref<*xf64>
// CHECK-NEXT:     "gpu.host_register"(%5) : (memref<*xf64>) -> ()
// CHECK-NEXT:     %6 = "memref.subview"(%2) {"static_offsets" = array<i64: 3, 3, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %9 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %10 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %11 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%7, %7, %7, %9, %10, %11, %8, %8, %8) ({
// CHECK-NEXT:     ^1(%12 : index, %13 : index, %14 : index):
// CHECK-NEXT:       %15 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %16 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %17 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %18 = "arith.addi"(%14, %15) : (index, index) -> index
// CHECK-NEXT:       %19 = "arith.addi"(%13, %16) : (index, index) -> index
// CHECK-NEXT:       %20 = "arith.addi"(%12, %17) : (index, index) -> index
// CHECK-NEXT:       %21 = "memref.load"(%6, %18, %19, %20) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %22 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %23 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %24 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:       %25 = "arith.addi"(%14, %22) : (index, index) -> index
// CHECK-NEXT:       %26 = "arith.addi"(%13, %23) : (index, index) -> index
// CHECK-NEXT:       %27 = "arith.addi"(%12, %24) : (index, index) -> index
// CHECK-NEXT:       %28 = "memref.load"(%6, %25, %26, %27) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %29 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %30 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:       %31 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %32 = "arith.addi"(%14, %29) : (index, index) -> index
// CHECK-NEXT:       %33 = "arith.addi"(%13, %30) : (index, index) -> index
// CHECK-NEXT:       %34 = "arith.addi"(%12, %31) : (index, index) -> index
// CHECK-NEXT:       %35 = "memref.load"(%6, %32, %33, %34) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %36 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %37 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %38 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %39 = "arith.addi"(%14, %36) : (index, index) -> index
// CHECK-NEXT:       %40 = "arith.addi"(%13, %37) : (index, index) -> index
// CHECK-NEXT:       %41 = "arith.addi"(%12, %38) : (index, index) -> index
// CHECK-NEXT:       %42 = "memref.load"(%6, %39, %40, %41) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %43 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %44 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %45 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %46 = "arith.addi"(%14, %43) : (index, index) -> index
// CHECK-NEXT:       %47 = "arith.addi"(%13, %44) : (index, index) -> index
// CHECK-NEXT:       %48 = "arith.addi"(%12, %45) : (index, index) -> index
// CHECK-NEXT:       %49 = "memref.load"(%6, %46, %47, %48) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %50 = "arith.addf"(%21, %28) : (f64, f64) -> f64
// CHECK-NEXT:       %51 = "arith.addf"(%35, %42) : (f64, f64) -> f64
// CHECK-NEXT:       %52 = "arith.addf"(%50, %51) : (f64, f64) -> f64
// CHECK-NEXT:       %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %53 = "arith.mulf"(%49, %cst) : (f64, f64) -> f64
// CHECK-NEXT:       %54 = "arith.addf"(%53, %52) : (f64, f64) -> f64
// CHECK-NEXT:       %55 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %56 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %57 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %58 = "arith.addi"(%14, %55) : (index, index) -> index
// CHECK-NEXT:       %59 = "arith.addi"(%13, %56) : (index, index) -> index
// CHECK-NEXT:       %60 = "arith.addi"(%12, %57) : (index, index) -> index
// CHECK-NEXT:       "memref.store"(%54, %4, %58, %59, %60) : (f64, memref<72x72x72xf64>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
// CHECK-NEXT: }) : () -> ()



