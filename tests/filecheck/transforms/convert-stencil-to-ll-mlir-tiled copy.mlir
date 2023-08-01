// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir{tile-sizes=16,24,32} | filecheck %s

builtin.module {
// CHECK: builtin.module {

  // The pass used to crash on external function, just regression-testing this here.
  func.func @external(!stencil.field<?xf64>) -> ()
  // CHECK: func.func @external(memref<?xf64>) -> ()

  func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = "stencil.apply"(%0) ({
    ^0(%4 : f64):
      %5 = arith.constant 1.0 : f64
      %6 = arith.addf %4, %5 : f64
      "stencil.return"(%6) : (f64) -> ()
    }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<1, 2, 3>, "ub" = #stencil.index<65, 66, 63>} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
    func.return
  }
  // CHECK:      func.func @stencil_init_float(%0 : f64, %1 : memref<?x?x?xf64>) {
  // CHECK-NEXT:   %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
  // CHECK-NEXT:   %3 = "memref.subview"(%2) {"static_offsets" = array<i64: 3, 3, 3>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>
  // CHECK-NEXT:   %4 = arith.constant 1 : index
  // CHECK-NEXT:   %5 = arith.constant 2 : index
  // CHECK-NEXT:   %6 = arith.constant 3 : index
  // CHECK-NEXT:   %7 = arith.constant 1 : index
  // CHECK-NEXT:   %8 = arith.constant 65 : index
  // CHECK-NEXT:   %9 = arith.constant 66 : index
  // CHECK-NEXT:   %10 = arith.constant 63 : index
  // CHECK-NEXT:   "scf.parallel"(%4, %8, %7) ({
  // CHECK-NEXT:   ^0(%11 : index):
  // CHECK-NEXT:     "scf.for"(%5, %9, %7) ({
  // CHECK-NEXT:     ^1(%12 : index):
  // CHECK-NEXT:       "scf.for"(%6, %10, %7) ({
  // CHECK-NEXT:       ^2(%13 : index):
  // CHECK-NEXT:         %14 = arith.constant 1.000000e+00 : f64
  // CHECK-NEXT:         %15 = arith.addf %0, %14 : f64
  // CHECK-NEXT:         "memref.store"(%15, %3, %11, %12, %13) : (f64, memref<64x64x60xf64, strided<[4900, 70, 1], offset: 14913>>, index, index, index) -> ()
  // CHECK-NEXT:         "scf.yield"() : () -> ()
  // CHECK-NEXT:       }) : (index, index, index) -> ()
  // CHECK-NEXT:       "scf.yield"() : () -> ()
  // CHECK-NEXT:     }) : (index, index, index) -> ()
  // CHECK-NEXT:     "scf.yield"() : () -> ()
  // CHECK-NEXT:   }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

}
// CHECK-NEXT: }
