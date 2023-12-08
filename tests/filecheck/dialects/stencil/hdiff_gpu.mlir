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
      %cst = arith.constant -4.0 : f64
      %18 = "arith.mulf"(%14, %cst) : (f64, f64) -> f64
      %19 = "arith.addf"(%18, %17) : (f64, f64) -> f64
      "stencil.return"(%19, %19, %19, %19) <{"unroll" = #stencil.index<2,2,1>}> : (f64, f64, f64, f64) -> ()
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
// CHECK-NEXT:     %6 = "memref.subview"(%4) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:     %7 = "memref.subview"(%2) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 66, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:     %8 = arith.constant 0 : index
// CHECK-NEXT:     %9 = arith.constant 0 : index
// CHECK-NEXT:     %10 = arith.constant 0 : index
// CHECK-NEXT:     %11 = arith.constant 0 : index
// CHECK-NEXT:     %12 = arith.constant 1 : index
// CHECK-NEXT:     %13 = arith.constant 64 : index
// CHECK-NEXT:     %14 = arith.constant 64 : index
// CHECK-NEXT:     %15 = arith.constant 64 : index
// CHECK-NEXT:     "scf.parallel"(%11, %10, %8, %15, %14, %13, %12, %12, %12) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:     ^0(%16 : index, %17 : index, %18 : index):
// CHECK-NEXT:       %19 = arith.constant -1 : index
// CHECK-NEXT:       %20 = arith.addi %18, %19 : index
// CHECK-NEXT:       %21 = memref.load %7[%20, %17, %16] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:       %22 = arith.constant 1 : index
// CHECK-NEXT:       %23 = arith.addi %18, %22 : index
// CHECK-NEXT:       %24 = memref.load %7[%23, %17, %16] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:       %25 = arith.constant 1 : index
// CHECK-NEXT:       %26 = arith.addi %17, %25 : index
// CHECK-NEXT:       %27 = memref.load %7[%18, %26, %16] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:       %28 = arith.constant -1 : index
// CHECK-NEXT:       %29 = arith.addi %17, %28 : index
// CHECK-NEXT:       %30 = memref.load %7[%18, %29, %16] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:       %31 = memref.load %7[%18, %17, %16] : memref<66x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:       %32 = arith.addf %21, %24 : f64
// CHECK-NEXT:       %33 = arith.addf %27, %30 : f64
// CHECK-NEXT:       %34 = arith.addf %32, %33 : f64
// CHECK-NEXT:       %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:       %35 = arith.mulf %31, %cst : f64
// CHECK-NEXT:       %36 = arith.addf %35, %34 : f64
// CHECK-NEXT:       memref.store %36, %6[%18, %17, %16] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
