"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "test.op"() : () -> i32
  %2 = "arith.addi"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %3 = "arith.addi"(%0, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %4 = "arith.addi"(%0, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %5 = "arith.addi"(%0, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %6 = "arith.addi"(%0, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "test.op"(%0) : (i32) -> ()
}) : () -> ()
