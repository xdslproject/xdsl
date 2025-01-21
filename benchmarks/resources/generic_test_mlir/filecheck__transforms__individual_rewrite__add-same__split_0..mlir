"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
