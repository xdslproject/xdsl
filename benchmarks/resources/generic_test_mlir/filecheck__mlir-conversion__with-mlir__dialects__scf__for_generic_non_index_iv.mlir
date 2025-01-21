"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %2 = "arith.constant"() <{value = 7 : i32}> : () -> i32
  %3 = "arith.constant"() <{value = 36 : i32}> : () -> i32
  %4 = "scf.for"(%0, %1, %2, %3) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %5 = "arith.addi"(%arg2, %arg1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "scf.yield"(%5) : (i32) -> ()
  }) : (i32, i32, i32, i32) -> i32
  "scf.for"(%0, %1, %2) ({
  ^bb0(%arg0: i32):
    "scf.yield"() : () -> ()
  }) : (i32, i32, i32) -> ()
}) : () -> ()
