"builtin.module"() ({
  %0:3 = "test.op"() : () -> (i32, i64, i32)
  "test.op"(%0#1, %0#0) : (i64, i32) -> ()
}) : () -> ()
