"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  "scf.for"(%0, %0, %0) ({
  ^bb0(%arg0: index):
    "unregistered_op"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()
