"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "arith.constant"() <{value = 1000 : index}> : () -> index
  %2 = "arith.constant"() <{value = 3 : index}> : () -> index
  %3 = "arith.constant"() <{value = 0 : index}> : () -> index
  %4 = "arith.constant"() <{value = 1000 : index}> : () -> index
  "scf.parallel"(%0, %3, %1, %4, %2) <{operandSegmentSizes = array<i32: 2, 2, 1, 0>}> ({
  ^bb0(%arg0: index):
    "scf.reduce"() : () -> ()
  }) : (index, index, index, index, index) -> ()
}) : () -> ()
