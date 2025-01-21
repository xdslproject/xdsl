"builtin.module"() ({
  %0 = "test.op"(%1) : (i32) -> i32
  %1 = "test.termop"(%0) : (i32) -> i32
}) : () -> ()
