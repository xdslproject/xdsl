// RUN: xdsl-opt %s -p stencil-shape-inference,convert-stencil-to-ll-mlir{target=gpu} | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>):
    %3 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %6 = "stencil.load"(%3) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<?x?x?xf64>
    %8 = "stencil.apply"(%6) ({
    ^1(%9 : !stencil.temp<?x?x?xf64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<?x?x?xf64>) -> f64
      %15 = "arith.addf"(%10, %11) : (f64, f64) -> f64
      %16 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %17 = "arith.addf"(%15, %16) : (f64, f64) -> f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %18 = "arith.mulf"(%14, %cst) : (f64, f64) -> f64
      %19 = "arith.addf"(%18, %17) : (f64, f64) -> f64
      "stencil.return"(%19) : (f64) -> ()
    }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    "stencil.store"(%8, %4) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<?x?x?xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    func.return
  }) {"function_type" = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @stencil_hdiff(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>) {
// CHECK-NEXT:     %2 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %3 = "memref.cast"(%2) : (memref<72x72x72xf64>) -> memref<*xf64>
// CHECK-NEXT:     "gpu.host_register"(%3) : (memref<*xf64>) -> ()
// CHECK-NEXT:     %4 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %5 = "memref.cast"(%4) : (memref<72x72x72xf64>) -> memref<*xf64>
// CHECK-NEXT:     "gpu.host_register"(%5) : (memref<*xf64>) -> ()
// CHECK-NEXT:     %6 = "memref.subview"(%4) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:     %7 = "memref.subview"(%2) {"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %9 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %10 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %11 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %12 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %13 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %14 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %15 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%11, %10, %8, %15, %14, %13, %12, %12, %12) ({
// CHECK-NEXT:     ^0(%16 : index, %17 : index, %18 : index):
// CHECK-NEXT:       %19 = "arith.constant"() {"value" = -1 : index} : () -> index
// CHECK-NEXT:       %20 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %21 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %22 = arith.addi %18, %19 : index
// CHECK-NEXT:       %23 = arith.addi %17, %20 : index
// CHECK-NEXT:       %24 = arith.addi %16, %21 : index
// CHECK-NEXT:       %25 = "memref.load"(%7, %22, %23, %24) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:       %26 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %27 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %28 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %29 = arith.addi %18, %26 : index
// CHECK-NEXT:       %30 = arith.addi %17, %27 : index
// CHECK-NEXT:       %31 = arith.addi %16, %28 : index
// CHECK-NEXT:       %32 = "memref.load"(%7, %29, %30, %31) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:       %33 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %34 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %35 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %36 = arith.addi %18, %33 : index
// CHECK-NEXT:       %37 = arith.addi %17, %34 : index
// CHECK-NEXT:       %38 = arith.addi %16, %35 : index
// CHECK-NEXT:       %39 = "memref.load"(%7, %36, %37, %38) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:       %40 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %41 = "arith.constant"() {"value" = -1 : index} : () -> index
// CHECK-NEXT:       %42 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %43 = arith.addi %18, %40 : index
// CHECK-NEXT:       %44 = arith.addi %17, %41 : index
// CHECK-NEXT:       %45 = arith.addi %16, %42 : index
// CHECK-NEXT:       %46 = "memref.load"(%7, %43, %44, %45) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:       %47 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %48 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %49 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %50 = arith.addi %18, %47 : index
// CHECK-NEXT:       %51 = arith.addi %17, %48 : index
// CHECK-NEXT:       %52 = arith.addi %16, %49 : index
// CHECK-NEXT:       %53 = "memref.load"(%7, %50, %51, %52) : (memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> f64
// CHECK-NEXT:       %54 = arith.addf %25, %32 : f64
// CHECK-NEXT:       %55 = arith.addf %39, %46 : f64
// CHECK-NEXT:       %56 = arith.addf %54, %55 : f64
// CHECK-NEXT:       %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %57 = arith.mulf %53, %cst : f64
// CHECK-NEXT:       %58 = arith.addf %57, %56 : f64
// CHECK-NEXT:       "memref.store"(%58, %6, %18, %17, %16) : (f64, memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
