// RUN: xdsl-opt %s --parsing-diagnostics | filecheck %s

"builtin.module"() ({

  %0:1 = "test.op"() : () -> (i32, i64, i32)
  // CHECK: Operation has 3 results, but was given 1 to bind.

}) : () -> ()
