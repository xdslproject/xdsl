"builtin.module"() ({
  "func.func"() <{function_type = (index, index, index, index) -> (), sym_name = "parallel_loop_4d"}> ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "arith.constant"() <{value = 4 : index}> : () -> index
    "scf.parallel"(%0, %0, %0, %0, %arg0, %arg1, %arg2, %arg3, %2, %2, %2, %2) <{operandSegmentSizes = array<i32: 4, 4, 4, 0>}> ({
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
      "scf.parallel"(%0, %0, %0, %0, %2, %2, %2, %2, %1, %1, %1, %1) <{operandSegmentSizes = array<i32: 4, 4, 4, 0>}> ({
      ^bb0(%arg8: index, %arg9: index, %arg10: index, %arg11: index):
        "scf.parallel"(%0, %0, %0, %0, %2, %2, %2, %2, %1, %1, %1, %1) <{operandSegmentSizes = array<i32: 4, 4, 4, 0>}> ({
        ^bb0(%arg12: index, %arg13: index, %arg14: index, %arg15: index):
          "scf.reduce"() : () -> ()
        }) : (index, index, index, index, index, index, index, index, index, index, index, index) -> ()
        "scf.reduce"() : () -> ()
      }) : (index, index, index, index, index, index, index, index, index, index, index, index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index, index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
