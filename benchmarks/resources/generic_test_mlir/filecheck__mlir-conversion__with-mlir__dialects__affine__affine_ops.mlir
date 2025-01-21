"builtin.module"() ({
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (256)>}> ({
  ^bb0(%arg6: index):
    "affine.yield"() : () -> ()
  }) : () -> ()
  %0 = "test.op"() : () -> i32
  %1 = "affine.for"(%0) <{lowerBoundMap = affine_map<() -> (-10)>, operandSegmentSizes = array<i32: 0, 0, 1>, step = 1 : index, upperBoundMap = affine_map<() -> (10)>}> ({
  ^bb0(%arg4: index, %arg5: i32):
    %14 = "test.op"() : () -> i32
    "affine.yield"(%14) : (i32) -> ()
  }) : (i32) -> i32
  %2 = "test.op"() : () -> index
  %3 = "test.op"() : () -> index
  %4 = "affine.for"(%2, %3, %0) <{lowerBoundMap = affine_map<(d0) -> (d0)>, operandSegmentSizes = array<i32: 1, 1, 1>, step = 1 : index, upperBoundMap = affine_map<()[s0] -> (s0)>}> ({
  ^bb0(%arg2: index, %arg3: i32):
    %13 = "test.op"() : () -> i32
    "affine.yield"(%13) : (i32) -> ()
  }) : (index, index, i32) -> i32
  "affine.parallel"(%3) <{lowerBoundsGroups = dense<1> : vector<1xi32>, lowerBoundsMap = affine_map<() -> (0)>, reductions = [], steps = [1], upperBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = affine_map<()[s0] -> (s0)>}> ({
  ^bb0(%arg1: index):
    "affine.yield"() : () -> ()
  }) : (index) -> ()
  %5 = "test.op"() : () -> memref<2x3xf64>
  %6 = "test.op"() : () -> f64
  "affine.store"(%6, %5) <{map = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()
  %7 = "test.op"() : () -> index
  %8 = "affine.apply"(%7, %7) <{map = affine_map<(d0)[s0] -> (d0 + s0 * 42 - 1)>}> : (index, index) -> index
  %9 = "affine.min"(%7) <{map = affine_map<(d0) -> (d0 + 41, d0)>}> : (index) -> index
  %10 = "affine.load"(%5, %7, %7) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64
  "func.func"() <{function_type = () -> (), sym_name = "empty"}> ({
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (10)>}> ({
    ^bb0(%arg0: index):
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
      "affine.yield"() : () -> ()
    }, {
    }) : () -> ()
    "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
      "affine.yield"() : () -> ()
    }, {
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> f32, sym_name = "affine_if"}> ({
    %11 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %12 = "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
      "affine.yield"(%11) : (f32) -> ()
    }, {
      "affine.yield"(%11) : (f32) -> ()
    }) : () -> f32
    "func.return"(%12) : (f32) -> ()
  }) : () -> ()
}) : () -> ()
