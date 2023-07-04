// RUN: xdsl-opt %s --print-op-generic | filecheck %s
// CHECK: module
"builtin.module"() ({
  "func.func"() ({
    // CHECK: ^{{.*}}(%{{.*}}: i32):
    ^bb0(%arg: i32 loc(unknown)):
      // CHECK: "test.op"
      %0 = "test.op"() : () -> !test.type<"int"> loc(unknown)
      "test.op"() : () -> () loc(unknown)
      "func.return"() : () -> () loc(unknown)
  }) {"function_type" = (i32) -> (), "sym_name" = "builtin"} : () -> ()
}) : () -> ()
