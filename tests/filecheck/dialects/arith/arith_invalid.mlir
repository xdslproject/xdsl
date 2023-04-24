// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %lhs, %rhs = "test.op"() : () -> (i32, i64)
  %res = "arith.addi"(%lhs, %rhs) : (i32, i64) -> i32

  // CHECK: expect all input and result types to be equal
}) : () -> ()
