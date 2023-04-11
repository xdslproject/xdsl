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

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>):
// CHECK-NEXT:     %2 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %3 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %4 = "memref.subview"(%3) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:     %5 = "memref.subview"(%2) {"static_offsets" = array<i64: 3, 3, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %9 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %10 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%6, %8, %7) ({
// CHECK-NEXT:     ^1(%11 : index):
// CHECK-NEXT:       "scf.for"(%6, %9, %7) ({
// CHECK-NEXT:       ^2(%12 : index):
// CHECK-NEXT:         "scf.for"(%6, %10, %7) ({
// CHECK-NEXT:         ^3(%13 : index):
// CHECK-NEXT:           %14 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %15 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %16 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %17 = "arith.addi"(%11, %14) : (index, index) -> index
// CHECK-NEXT:           %18 = "arith.addi"(%12, %15) : (index, index) -> index
// CHECK-NEXT:           %19 = "arith.addi"(%13, %16) : (index, index) -> index
// CHECK-NEXT:           %20 = "memref.load"(%5, %17, %18, %19) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %21 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:           %22 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %23 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %24 = "arith.addi"(%11, %21) : (index, index) -> index
// CHECK-NEXT:           %25 = "arith.addi"(%12, %22) : (index, index) -> index
// CHECK-NEXT:           %26 = "arith.addi"(%13, %23) : (index, index) -> index
// CHECK-NEXT:           %27 = "memref.load"(%5, %24, %25, %26) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %28 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %29 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:           %30 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %31 = "arith.addi"(%11, %28) : (index, index) -> index
// CHECK-NEXT:           %32 = "arith.addi"(%12, %29) : (index, index) -> index
// CHECK-NEXT:           %33 = "arith.addi"(%13, %30) : (index, index) -> index
// CHECK-NEXT:           %34 = "memref.load"(%5, %31, %32, %33) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %35 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %36 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %37 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %38 = "arith.addi"(%11, %35) : (index, index) -> index
// CHECK-NEXT:           %39 = "arith.addi"(%12, %36) : (index, index) -> index
// CHECK-NEXT:           %40 = "arith.addi"(%13, %37) : (index, index) -> index
// CHECK-NEXT:           %41 = "memref.load"(%5, %38, %39, %40) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %42 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %43 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %44 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %45 = "arith.addi"(%11, %42) : (index, index) -> index
// CHECK-NEXT:           %46 = "arith.addi"(%12, %43) : (index, index) -> index
// CHECK-NEXT:           %47 = "arith.addi"(%13, %44) : (index, index) -> index
// CHECK-NEXT:           %48 = "memref.load"(%5, %45, %46, %47) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %49 = "arith.addf"(%20, %27) : (f64, f64) -> f64
// CHECK-NEXT:           %50 = "arith.addf"(%34, %41) : (f64, f64) -> f64
// CHECK-NEXT:           %51 = "arith.addf"(%49, %50) : (f64, f64) -> f64
// CHECK-NEXT:           %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:           %52 = "arith.mulf"(%48, %cst) : (f64, f64) -> f64
// CHECK-NEXT:           %53 = "arith.addf"(%52, %51) : (f64, f64) -> f64
// CHECK-NEXT:           "memref.store"(%53, %4, %11, %12, %13) : (f64, memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
// CHECK-NEXT: }) : () -> ()
