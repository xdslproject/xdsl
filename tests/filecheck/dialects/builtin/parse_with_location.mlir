// RUN: xdsl-opt %s --print-op-generic | filecheck %s
// CHECK: module
"builtin.module"() ({
  // CHECK: ^{{.*}}(%{{.*}}: i32):
  ^bb0(%arg: i32 loc(unknown)):
    // CHECK: "test.op"() : () -> !test.type<"int">
    %0 = "test.op"() : () -> !test.type<"int"> loc(unknown)
    // CHECK: "test.op"() : () -> ()
    "test.op"() : () -> () loc(unknown)
}) : () -> ()
