"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "arith.constant"() <{value = 1000 : index}> : () -> index
  %2 = "arith.constant"() <{value = 3 : index}> : () -> index
  %3 = "arith.constant"() <{value = 10 : i32}> : () -> i32
  %4 = "arith.constant"() <{value = 100 : i32}> : () -> i32
  %5 = "scf.parallel"(%0, %1, %2, %3) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> ({
  ^bb0(%arg0: index):
    "scf.reduce"(%4) ({
    ^bb0(%arg1: i32, %arg2: i32):
      %6 = "arith.addi"(%arg1, %arg2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.reduce.return"(%6) : (i32) -> ()
    }) : (i32) -> ()
  }) : (index, index, index, i32) -> i32
}) : () -> ()
