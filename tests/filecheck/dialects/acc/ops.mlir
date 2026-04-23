// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func @acc_parallel_empty() {
    "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  // CHECK:      "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  // CHECK-NEXT:   acc.yield
  // CHECK-NEXT: }) : () -> ()

  func.func @acc_parallel_yield_operands(%arg0: memref<10xf32>) {
    "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"(%arg0) : (memref<10xf32>) -> ()
    }) : () -> ()
    func.return
  }
  // CHECK:      "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  // CHECK-NEXT:   acc.yield %{{.*}} : memref<10xf32>
  // CHECK-NEXT: }) : () -> ()
}
