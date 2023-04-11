// RUN: xdsl-opt %s -t mlir --allow-unregistered-dialect --parsing-diagnostics | filecheck %s

"builtin.module"() ({

  %0:1 = "test.test"() : () -> (i32, i64, i32)
  // CHECK: Operation has 3 results, but were given 1 to bind.

}) : () -> ()
