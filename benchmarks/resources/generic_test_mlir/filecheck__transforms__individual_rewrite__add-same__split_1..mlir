"builtin.module"() ({
  %0 = "test.op"() : () -> i1
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
}) : () -> ()
