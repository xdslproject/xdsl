// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

module {
  func.func public @my_func() {
    return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @my_func() {
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
