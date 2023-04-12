// RUN: xdsl-opt %s -t mlir -p stencil-shape-inference,convert-stencil-to-ll-mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %2 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
    %3 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %4 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %5 = "stencil.cast"(%2) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %6 = "stencil.load"(%3) : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    %7, %8 = "stencil.apply"(%6) ({
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
      "stencil.return"(%19, %18) : (f64, f64) -> ()
    }) : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>)
    "stencil.store"(%7, %4) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
    "stencil.store"(%8, %5) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>, %2 : memref<?x?x?xf64>):
// CHECK-NEXT:     %3 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %4 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %5 = "memref.subview"(%4) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:     %6 = "memref.cast"(%2) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %7 = "memref.subview"(%6) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:     %8 = "memref.subview"(%3) {"static_offsets" = array<i64: 3, 3, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>
// CHECK-NEXT:     %9 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %10 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %11 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %12 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %13 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%9, %11, %10) ({
// CHECK-NEXT:     ^1(%14 : index):
// CHECK-NEXT:       "scf.for"(%9, %12, %10) ({
// CHECK-NEXT:       ^2(%15 : index):
// CHECK-NEXT:         "scf.for"(%9, %13, %10) ({
// CHECK-NEXT:         ^3(%16 : index):
// CHECK-NEXT:           %17 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %18 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %19 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %20 = "arith.addi"(%14, %17) : (index, index) -> index
// CHECK-NEXT:           %21 = "arith.addi"(%15, %18) : (index, index) -> index
// CHECK-NEXT:           %22 = "arith.addi"(%16, %19) : (index, index) -> index
// CHECK-NEXT:           %23 = "memref.load"(%8, %20, %21, %22) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %24 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:           %25 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %26 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %27 = "arith.addi"(%14, %24) : (index, index) -> index
// CHECK-NEXT:           %28 = "arith.addi"(%15, %25) : (index, index) -> index
// CHECK-NEXT:           %29 = "arith.addi"(%16, %26) : (index, index) -> index
// CHECK-NEXT:           %30 = "memref.load"(%8, %27, %28, %29) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %31 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %32 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:           %33 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %34 = "arith.addi"(%14, %31) : (index, index) -> index
// CHECK-NEXT:           %35 = "arith.addi"(%15, %32) : (index, index) -> index
// CHECK-NEXT:           %36 = "arith.addi"(%16, %33) : (index, index) -> index
// CHECK-NEXT:           %37 = "memref.load"(%8, %34, %35, %36) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %38 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %39 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %40 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %41 = "arith.addi"(%14, %38) : (index, index) -> index
// CHECK-NEXT:           %42 = "arith.addi"(%15, %39) : (index, index) -> index
// CHECK-NEXT:           %43 = "arith.addi"(%16, %40) : (index, index) -> index
// CHECK-NEXT:           %44 = "memref.load"(%8, %41, %42, %43) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %45 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %46 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:           %47 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %48 = "arith.addi"(%14, %45) : (index, index) -> index
// CHECK-NEXT:           %49 = "arith.addi"(%15, %46) : (index, index) -> index
// CHECK-NEXT:           %50 = "arith.addi"(%16, %47) : (index, index) -> index
// CHECK-NEXT:           %51 = "memref.load"(%8, %48, %49, %50) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 15772>>, index, index, index) -> f64
// CHECK-NEXT:           %52 = "arith.addf"(%23, %30) : (f64, f64) -> f64
// CHECK-NEXT:           %53 = "arith.addf"(%37, %44) : (f64, f64) -> f64
// CHECK-NEXT:           %54 = "arith.addf"(%52, %53) : (f64, f64) -> f64
// CHECK-NEXT:           %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:           %55 = "arith.mulf"(%51, %cst) : (f64, f64) -> f64
// CHECK-NEXT:           %56 = "arith.addf"(%55, %54) : (f64, f64) -> f64
// CHECK-NEXT:           "memref.store"(%56, %5, %14, %15, %16) : (f64, memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> ()
// CHECK-NEXT:           "memref.store"(%55, %7, %14, %15, %16) : (f64, memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
// CHECK-NEXT: }) : () -> ()
