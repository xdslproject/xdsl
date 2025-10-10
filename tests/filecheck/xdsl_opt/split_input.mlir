// RUN: xdsl-opt --split-input-file %s | xdsl-opt --split-input-file --print-op-generic | filecheck %s

"builtin.module"() ({
  ^bb0:
// CHECK: "builtin.module"() ({
}) : () -> ()

// -----
"builtin.module"() ({
  "test.op"() : () -> ()
// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "test.op"() : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  %x = "test.op"() : () -> i1
// CHECK: "builtin.module"() ({
// CHECK:   %x = "test.op"() : () -> i1
// CHECK-NOT:   %x = "test.op"() : () -> i2
// CHECK-NOT:   "test.op"() : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  %x = "test.op"() : () -> i2
// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %x = "test.op"() : () -> i2
// CHECK-NOT:   %x = "test.op"() : () -> i1
// CHECK-NOT:   "test.op"() : () -> ()
}) : () -> ()
