"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "tile_nested_innermost"}> ({
    %0 = "arith.constant"() <{value = 2 : index}> : () -> index
    %1 = "arith.constant"() <{value = 0 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.parallel"(%1, %1, %0, %0, %2, %2) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg2: index, %arg3: index):
      "scf.parallel"(%1, %1, %0, %0, %2, %2) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
      ^bb0(%arg4: index, %arg5: index):
        "scf.reduce"() : () -> ()
      }) : (index, index, index, index, index, index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index) -> ()
    "scf.parallel"(%1, %1, %0, %0, %2, %2) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg0: index, %arg1: index):
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
