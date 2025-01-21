// RUN: xdsl-opt %s | xdsl-opt  | filecheck %s

"builtin.module"() ({

  %0:3 = "test.op"() : () -> (i32, i64, i32)
  "test.op"(%0#1, %0#0) : (i64, i32) -> ()

  // CHECK: %0, %1, %2 = "test.op"() : () -> (i32, i64, i32)
  // CHECK: "test.op"(%1, %0) : (i64, i32) -> ()


}) : () -> ()
