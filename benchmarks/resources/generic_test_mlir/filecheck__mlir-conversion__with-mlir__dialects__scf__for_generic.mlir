"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "arith.constant"() <{value = 42 : index}> : () -> index
  %2 = "arith.constant"() <{value = 7 : index}> : () -> index
  %3 = "arith.constant"() <{value = 36 : index}> : () -> index
  %4 = "scf.for"(%0, %1, %2, %3) ({
  ^bb0(%arg1: index, %arg2: index):
    %5 = "arith.addi"(%arg2, %arg1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    "scf.yield"(%5) : (index) -> ()
  }) : (index, index, index, index) -> index
  "scf.for"(%0, %1, %2) ({
  ^bb0(%arg0: index):
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
