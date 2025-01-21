"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "tile_partial"}> ({
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    %4 = "arith.constant"() <{value = 1 : index}> : () -> index
    %5 = "arith.constant"() <{value = 64 : index}> : () -> index
    "scf.parallel"(%3, %3, %3, %5, %5, %5, %4, %4, %4) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "tile_partial_1d"}> ({
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "arith.constant"() <{value = 64 : index}> : () -> index
    "scf.parallel"(%0, %2, %1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb0(%arg0: index):
      "scf.reduce"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
