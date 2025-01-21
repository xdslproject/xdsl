"builtin.module"() ({
  %0:2 = "test.op"() : () -> (f32, memref<2x3xf32>)
  "affine.store"(%0#0, %0#1) <{map = affine_map<() -> (1, 2)>}> : (f32, memref<2x3xf32>) -> ()
  %1 = "affine.load"(%0#1) <{map = affine_map<() -> (1, 2)>}> : (memref<2x3xf32>) -> f32
  %2 = "affine.for"(%1) <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 1>, step = 1 : index, upperBoundMap = affine_map<() -> (2)>}> ({
  ^bb0(%arg0: index, %arg1: f32):
    %5 = "affine.for"(%arg1) <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 1>, step = 1 : index, upperBoundMap = affine_map<() -> (3)>}> ({
    ^bb0(%arg2: index, %arg3: f32):
      %6 = "affine.load"(%0#1, %arg0, %arg2) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf32>, index, index) -> f32
      %7 = "test.op"(%arg3, %6) : (f32, f32) -> f32
      "affine.yield"(%7) : (f32) -> ()
    }) : (f32) -> f32
    "affine.yield"(%5) : (f32) -> ()
  }) : (f32) -> f32
  %3:2 = "test.op"() : () -> (index, index)
  %4 = "affine.apply"(%3#0, %3#1) <{map = affine_map<(d0)[s0] -> (d0 + s0 * 42 - 1)>}> : (index, index) -> index
}) : () -> ()
