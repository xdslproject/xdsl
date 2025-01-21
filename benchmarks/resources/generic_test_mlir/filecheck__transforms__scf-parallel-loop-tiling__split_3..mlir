"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "tile_nested_in_non_ploop"}> ({
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "arith.constant"() <{value = 2 : index}> : () -> index
    "scf.for"(%0, %2, %1) ({
    ^bb0(%arg0: index):
      "scf.for"(%0, %2, %1) ({
      ^bb0(%arg1: index):
        "scf.parallel"(%0, %0, %2, %2, %1, %1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
        ^bb0(%arg2: index, %arg3: index):
          "scf.reduce"() : () -> ()
        }) : (index, index, index, index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
