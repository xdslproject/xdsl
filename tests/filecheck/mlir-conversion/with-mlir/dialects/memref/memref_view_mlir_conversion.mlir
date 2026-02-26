// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

func.func private @view_test() {
  %c0 = arith.constant 0 : index
  %M = arith.constant 64 : index
  %N = arith.constant 4 : index

  %src = memref.alloc() : memref<2048xi8>

  // View with dynamic offset and static sizes.
  %A = memref.view %src[%c0][] : memref<2048xi8> to memref<64x4xf32>

  // View with dynamic offset and dynamic sizes.
  %B = memref.view %src[%c0][%M, %N] : memref<2048xi8> to memref<?x?xf32>

  func.return
}

// CHECK: func.func private @view_test() {
// CHECK:   %{{.*}} = arith.constant 0 : index
// CHECK:   %{{.*}} = arith.constant 64 : index
// CHECK:   %{{.*}} = arith.constant 4 : index
// CHECK:   %[[SRC:.*]] = memref.alloc() : memref<2048xi8>
// CHECK:   %{{.*}} = memref.view %[[SRC]][%{{.*}}][] : memref<2048xi8> to memref<64x4xf32>
// CHECK:   %{{.*}} = memref.view %[[SRC]][%{{.*}}][%{{.*}}, %{{.*}}] : memref<2048xi8> to memref<?x?xf32>
// CHECK:   func.return
// CHECK: }
