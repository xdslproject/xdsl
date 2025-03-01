// RUN: xdsl-opt %s --split-input-file --print-op-generic --allow-unregistered-dialect | xdsl-opt --allow-unregistered-dialect | filecheck %s

// CHECK:        %0 = arith.constant 0 : index
// CHECK-NEXT:   scf.for %iv = %0 to %0 step %0 {
// CHECK-NEXT:       "unregistered_op"() : () -> ()
// CHECK-NEXT:   }

%0 = arith.constant 0 : index
scf.for %iv = %0 to %0 step %0 {
    "unregistered_op"() : () -> ()
    scf.yield
}

// -----

// CHECK:        %0 = arith.constant 0 : index
// CHECK-NEXT:   scf.for %iv = %0 to %0 step %0 {
// CHECK-NEXT:       "unregistered_op"() : () -> ()
// CHECK-NEXT:   }

%0 = arith.constant 0 : index
scf.for %iv = %0 to %0 step %0 {
    "unregistered_op"() : () -> ()
}
