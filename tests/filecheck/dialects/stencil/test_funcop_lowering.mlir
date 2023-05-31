// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir | filecheck %s

builtin.module {
  func.func @test_funcop_lowering(%0 : !stencil.field<?x?x?xf64>) {
    "func.return"() : () -> ()
  }

  func.func @test_funcop_lowering(%1 : !stencil.field<[-1,7]x[-1,7]xf64>) {
    "func.return"() : () -> ()
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @test_funcop_lowering(%0 : memref<?x?x?xf64>) {
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @test_funcop_lowering(%1 : memref<8x8xf64>) {
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
