// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s

builtin.module {
  func.func @acc_parallel_empty() {
    "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @acc_parallel_empty() {
// CHECK-NEXT:      "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
