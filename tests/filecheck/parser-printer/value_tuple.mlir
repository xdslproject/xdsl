// RUN: xdsl-opt %s --allow-unregistered-dialect | xdsl-opt --allow-unregistered-dialect  | filecheck %s

"builtin.module"() ({

  %0:3 = "test.test"() : () -> (i32, i64, i32)
  "test.test"(%0#1, %0#0) : (i64, i32) -> ()

  // CHECK: %0, %1, %2 = "test.test"() : () -> (i32, i64, i32)
  // CHECK: "test.test"(%1, %0) : (i64, i32) -> ()


}) : () -> ()
