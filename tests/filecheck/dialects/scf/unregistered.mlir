// RUN: xdsl-opt %s --print-op-generic --allow-unregistered-dialect | xdsl-opt --allow-unregistered-dialect | filecheck %s

// CHECK:      func.func @for_unregistered() {
// CHECK-NEXT:   %lb = arith.constant 0 : index
// CHECK-NEXT:   %ub = arith.constant 42 : index
// CHECK-NEXT:   %s = arith.constant 3 : index
// CHECK-NEXT:   scf.for %iv = %lb to %ub step %s {
// CHECK-NEXT:     "unregistered_op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

func.func @for_unregistered() {
  %lb = arith.constant 0 : index
  %ub = arith.constant 42 : index
  %s = arith.constant 3 : index
  scf.for %iv = %lb to %ub step %s {
    "unregistered_op"() : () -> ()
    scf.yield
  }
  func.return
}
