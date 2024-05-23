// RUN: xdsl-opt "%s" --verify-diagnostics | filecheck "%s"

"builtin.module"() ({

  %lhs, %rhs = "test.op"() : () -> (i32, i64)
  %res = "arith.addi"(%lhs, %rhs) : (i32, i64) -> i32

  // CHECK: attribute i32 expected from variable 'T', but got i64
}) : () -> ()
