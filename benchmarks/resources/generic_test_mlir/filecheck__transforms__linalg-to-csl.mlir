"builtin.module"() ({
  %0:5 = "test.op"() : () -> (memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>)
  "linalg.add"(%0#1, %0#2, %0#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg25: f32, %arg26: f32, %arg27: f32):
    %13 = "arith.addf"(%arg25, %arg26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%13) : (f32) -> ()
  }) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
  "linalg.sub"(%0#0, %0#3, %0#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg22: f32, %arg23: f32, %arg24: f32):
    %12 = "arith.subf"(%arg22, %arg23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%12) : (f32) -> ()
  }) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
  "linalg.mul"(%0#0, %0#4, %0#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg19: f32, %arg20: f32, %arg21: f32):
    %11 = "arith.mulf"(%arg19, %arg20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%11) : (f32) -> ()
  }) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
  %1:5 = "test.op"() : () -> (memref<16xf16>, memref<16xf16>, memref<16xf16>, memref<16xf16>, memref<16xf16>)
  "linalg.add"(%1#1, %1#2, %1#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg16: f16, %arg17: f16, %arg18: f16):
    %10 = "arith.addf"(%arg16, %arg17) <{fastmath = #arith.fastmath<none>}> : (f16, f16) -> f16
    "linalg.yield"(%10) : (f16) -> ()
  }) : (memref<16xf16>, memref<16xf16>, memref<16xf16>) -> ()
  "linalg.sub"(%1#0, %1#3, %1#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg13: f16, %arg14: f16, %arg15: f16):
    %9 = "arith.subf"(%arg13, %arg14) <{fastmath = #arith.fastmath<none>}> : (f16, f16) -> f16
    "linalg.yield"(%9) : (f16) -> ()
  }) : (memref<16xf16>, memref<16xf16>, memref<16xf16>) -> ()
  "linalg.mul"(%1#0, %1#4, %1#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg10: f16, %arg11: f16, %arg12: f16):
    %8 = "arith.mulf"(%arg10, %arg11) <{fastmath = #arith.fastmath<none>}> : (f16, f16) -> f16
    "linalg.yield"(%8) : (f16) -> ()
  }) : (memref<16xf16>, memref<16xf16>, memref<16xf16>) -> ()
  %2 = "arith.constant"() <{value = dense<1.123400e-01> : memref<16xf32>}> : () -> memref<16xf32>
  "linalg.add"(%0#0, %2, %0#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
    %7 = "arith.addf"(%arg7, %arg8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%7) : (f32) -> ()
  }) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
  "linalg.mul"(%2, %0#0, %0#0) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
    %6 = "arith.mulf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%6) : (f32) -> ()
  }) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
  %3 = "arith.constant"() <{value = dense<0x4D8EF3C2> : memref<16xf32>}> : () -> memref<16xf32>
  "linalg.generic"(%0#0, %3, %0#2, %0#0) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
    %4 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %5 = "arith.addf"(%4, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%5) : (f32) -> ()
  }) : (memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
}) : () -> ()
