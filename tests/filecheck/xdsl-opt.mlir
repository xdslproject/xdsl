// RUN: xdsl-opt -split-input-file %s | xdsl-opt -split-input-file | filecheck %s

"builtin.module"() ({
}) : () -> ()

// -----
"builtin.module"() ({
  "test.op"() : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  %x = "test.op"() : () -> i1
}) : () -> ()

// -----
"builtin.module"() ({
  %x = "test.op"() : () -> i2
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK: "test.op"() : () -> ()
// CHECK: }) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK:   %x = "test.op"() : () -> i1
// CHECK: }) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK:   %x = "test.op"() : () -> i2
// CHECK: }) : () -> ()

