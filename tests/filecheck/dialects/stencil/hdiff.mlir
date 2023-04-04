// RUN: xdsl-opt %s -t mlir -p stencil-shape-inference,convert-stencil-to-ll-mlir | filecheck %s


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
// CHECK-NEXT:     %3 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %4 = "memref.subview"(%2) {"static_offsets" = array<i64: 3, 3, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>
// CHECK-NEXT:     %5 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %9 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%5, %5, %5, %7, %8, %9, %6, %6, %6) ({
// CHECK-NEXT:     ^1(%10 : index, %11 : index, %12 : index):
// CHECK-NEXT:       %13 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %14 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %15 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %16 = "arith.addi"(%12, %13) : (index, index) -> index
// CHECK-NEXT:       %17 = "arith.addi"(%11, %14) : (index, index) -> index
// CHECK-NEXT:       %18 = "arith.addi"(%10, %15) : (index, index) -> index
// CHECK-NEXT:       %19 = "memref.load"(%4, %16, %17, %18) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %20 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %21 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %22 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:       %23 = "arith.addi"(%12, %20) : (index, index) -> index
// CHECK-NEXT:       %24 = "arith.addi"(%11, %21) : (index, index) -> index
// CHECK-NEXT:       %25 = "arith.addi"(%10, %22) : (index, index) -> index
// CHECK-NEXT:       %26 = "memref.load"(%4, %23, %24, %25) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %27 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %28 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:       %29 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %30 = "arith.addi"(%12, %27) : (index, index) -> index
// CHECK-NEXT:       %31 = "arith.addi"(%11, %28) : (index, index) -> index
// CHECK-NEXT:       %32 = "arith.addi"(%10, %29) : (index, index) -> index
// CHECK-NEXT:       %33 = "memref.load"(%4, %30, %31, %32) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %34 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %35 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %36 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %37 = "arith.addi"(%12, %34) : (index, index) -> index
// CHECK-NEXT:       %38 = "arith.addi"(%11, %35) : (index, index) -> index
// CHECK-NEXT:       %39 = "arith.addi"(%10, %36) : (index, index) -> index
// CHECK-NEXT:       %40 = "memref.load"(%4, %37, %38, %39) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %41 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %42 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %43 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %44 = "arith.addi"(%12, %41) : (index, index) -> index
// CHECK-NEXT:       %45 = "arith.addi"(%11, %42) : (index, index) -> index
// CHECK-NEXT:       %46 = "arith.addi"(%10, %43) : (index, index) -> index
// CHECK-NEXT:       %47 = "memref.load"(%4, %44, %45, %46) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:       %48 = "arith.addf"(%19, %26) : (f64, f64) -> f64
// CHECK-NEXT:       %49 = "arith.addf"(%33, %40) : (f64, f64) -> f64
// CHECK-NEXT:       %50 = "arith.addf"(%48, %49) : (f64, f64) -> f64
// CHECK-NEXT:       %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %51 = "arith.mulf"(%47, %cst) : (f64, f64) -> f64
// CHECK-NEXT:       %52 = "arith.addf"(%51, %50) : (f64, f64) -> f64
// CHECK-NEXT:       %53 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %54 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %55 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %56 = "arith.addi"(%12, %53) : (index, index) -> index
// CHECK-NEXT:       %57 = "arith.addi"(%11, %54) : (index, index) -> index
// CHECK-NEXT:       %58 = "arith.addi"(%10, %55) : (index, index) -> index
// CHECK-NEXT:       "memref.store"(%52, %3, %56, %57, %58) : (f64, memref<72x72x72xf64>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
// CHECK-NEXT: }) : () -> ()



