// RUN: xdsl-opt %s --print-op-generic | filecheck %s
// CHECK: module
"builtin.module"() ({
  "func.func"() ({
    // CHECK: ^{{.*}}(%{{.*}}: i32):
    ^bb0(%arg: i32 loc(unknown)):
      // CHECK: "arith.constant"
      %x1 = "arith.constant"() {"value" = 0 : i64} : () -> i64 loc(unknown)
      "func.return"() : () -> () loc(unknown)
  }) {"function_type" = (i32) -> (), "sym_name" = "builtin"} : () -> ()
}) : () -> ()
