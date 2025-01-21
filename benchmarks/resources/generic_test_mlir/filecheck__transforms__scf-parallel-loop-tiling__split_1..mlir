"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "static_loop_with_step"}> ({
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 3 : index}> : () -> index
    %2 = "arith.constant"() <{value = 22 : index}> : () -> index
    %3 = "arith.constant"() <{value = 24 : index}> : () -> index
    "scf.parallel"(%0, %0, %2, %3, %1, %1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg0: index, %arg1: index):
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
