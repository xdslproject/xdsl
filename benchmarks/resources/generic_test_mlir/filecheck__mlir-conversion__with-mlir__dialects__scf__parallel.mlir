"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "arith.constant"() <{value = 1000 : index}> : () -> index
  %2 = "arith.constant"() <{value = 3 : index}> : () -> index
  "scf.parallel"(%0, %1, %2) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%arg0: index):
    "scf.reduce"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
