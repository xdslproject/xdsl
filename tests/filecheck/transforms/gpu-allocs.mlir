// RUN: xdsl-opt %s -p memref-to-gpu | filecheck %s

func.func private @memref_test() {
  %12 = "test.op"() : () -> index
  %13 = "test.op"() : () -> index
  %14 = "test.op"() : () -> index
  %10 = memref.alloc() : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
  %15 = memref.alloc(%12) {"alignment" = 0} : memref<?xindex>
  %16 = memref.alloc(%12, %13, %14) {"alignment" = 0} : memref<?x?x?xindex>
  memref.dealloc %10 : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
  memref.dealloc %15 : memref<?xindex>
  memref.dealloc %16 : memref<?x?x?xindex>
  func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func private @memref_test() {
// CHECK-NEXT:      %0 = "test.op"() : () -> index
// CHECK-NEXT:      %1 = "test.op"() : () -> index
// CHECK-NEXT:      %2 = "test.op"() : () -> index
// CHECK-NEXT:      %3 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
// CHECK-NEXT:      %4 = "gpu.alloc"(%0) <{operandSegmentSizes = array<i32: 0, 1, 0>}> : (index) -> memref<?xindex>
// CHECK-NEXT:      %5 = "gpu.alloc"(%0, %1, %2) <{operandSegmentSizes = array<i32: 0, 3, 0>}> : (index, index, index) -> memref<?x?x?xindex>
// CHECK-NEXT:      "gpu.dealloc"(%3) {operandSegmentSizes = array<i32: 0, 1>} : (memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>) -> ()
// CHECK-NEXT:      "gpu.dealloc"(%4) {operandSegmentSizes = array<i32: 0, 1>} : (memref<?xindex>) -> ()
// CHECK-NEXT:      "gpu.dealloc"(%5) {operandSegmentSizes = array<i32: 0, 1>} : (memref<?x?x?xindex>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
