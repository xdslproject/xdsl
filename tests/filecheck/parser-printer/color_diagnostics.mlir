// RUN: xdsl-opt %s --syntax-highlight --verify-diagnostics | filecheck %s

%lhs, %rhs = "test.op"() : () -> (i32, i64)
// CHECK: [31m%res = "arith.addi"(%lhs, %rhs) <{overflowFlags = #arith.overflow<none>}> : (i32, i64) -> i32[0m
%res = "arith.addi"(%lhs, %rhs) : (i32, i64) -> i32
