"builtin.module"() ({
  "func.func"() <{function_type = (index, index, index, index, index, index) -> (), sym_name = "parallel"}> ({
  ^bb0(%arg25: index, %arg26: index, %arg27: index, %arg28: index, %arg29: index, %arg30: index):
    "scf.parallel"(%arg25, %arg26, %arg27, %arg28, %arg29, %arg30) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg31: index, %arg32: index):
      "test.op"(%arg31, %arg32) : (index, index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (index, index, index, index, index, index) -> (), sym_name = "nested_loops"}> ({
  ^bb0(%arg17: index, %arg18: index, %arg19: index, %arg20: index, %arg21: index, %arg22: index):
    "scf.parallel"(%arg17, %arg19, %arg21) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb0(%arg23: index):
      "scf.parallel"(%arg18, %arg20, %arg22) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
      ^bb0(%arg24: index):
        "test.op"(%arg23, %arg24) : (index, index) -> ()
        "scf.reduce"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (index, index, index, index, index, index) -> (), sym_name = "adjacent_loops"}> ({
  ^bb0(%arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index, %arg14: index):
    "scf.parallel"(%arg9, %arg11, %arg13) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb0(%arg16: index):
      "test.op"(%arg16) : (index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.parallel"(%arg10, %arg12, %arg14) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb0(%arg15: index):
      "test.op"(%arg15) : (index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (index, index, index, index, index) -> (), sym_name = "reduction1"}> ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
    %0 = "arith.constant"() <{value = 1 : index}> : () -> index
    %1 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %2 = "scf.parallel"(%arg0, %arg1, %arg2, %arg3, %arg4, %0, %1) <{operandSegmentSizes = array<i32: 2, 2, 2, 1>}> ({
    ^bb0(%arg5: index, %arg6: index):
      %3 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      "scf.reduce"(%3) ({
      ^bb0(%arg7: f32, %arg8: f32):
        %4 = "arith.addf"(%arg7, %arg8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "scf.reduce.return"(%4) : (f32) -> ()
      }) : (f32) -> ()
    }) : (index, index, index, index, index, index, f32) -> f32
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
