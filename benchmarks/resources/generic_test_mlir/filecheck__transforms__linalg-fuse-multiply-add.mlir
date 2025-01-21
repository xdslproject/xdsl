"builtin.module"() ({
  %0:4 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %1 = "arith.constant"() <{value = dense<0x4D8EF3C4> : tensor<8xf32>}> : () -> tensor<8xf32>
  %2 = "linalg.mul"(%0#0, %0#1, %0#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg12: f32, %arg13: f32, %arg14: f32):
    %11 = "arith.mulf"(%arg12, %arg13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%11) : (f32) -> ()
  }) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %3 = "linalg.mul"(%1, %0#1, %1) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
    %10 = "arith.mulf"(%arg9, %arg10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%10) : (f32) -> ()
  }) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %4 = "linalg.add"(%2, %0#2, %2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
    %9 = "arith.addf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%9) : (f32) -> ()
  }) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %5 = "linalg.add"(%3, %0#3, %3) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %8 = "arith.addf"(%arg3, %arg4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%8) : (f32) -> ()
  }) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %6 = "linalg.sub"(%3, %0#3, %3) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %7 = "arith.subf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%7) : (f32) -> ()
  }) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
}) : () -> ()
