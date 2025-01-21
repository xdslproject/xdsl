// RUN: xdsl-opt %s --allow-unregistered-dialect --parsing-diagnostics | filecheck %s

"builtin.module"() ({

  %0:3 = "test.op"() : () -> (i32, i64, i32)
  "test.op"(%0#3) : (i32) -> ()
  // CHECK: SSA value tuple index out of bounds. Tuple is of size 3 but tried to access element 3.

}) : () -> ()
