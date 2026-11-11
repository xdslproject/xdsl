// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

"test.region_branch_op"() ({
  "test.region_branch_terminator"() : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.region_branch_op"() ({
// CHECK-NEXT:     "test.region_branch_terminator"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }

// -----

"test.region_branch_op"() ({
  %0 = "test.region_branch_terminator"() : () -> i32
}) : () -> ()

// CHECK: Expected operation 'test.region_branch_terminator' to have zero results.

// -----

"test.region_branch_op"() ({
  "test.region_branch_terminator"() [^bb1] : () -> ()
^bb1:
  "test.region_branch_terminator"() : () -> ()
}) : () -> ()

// CHECK: Expected operation 'test.region_branch_terminator' to have zero successors.
