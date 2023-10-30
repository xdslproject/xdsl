// RUN: xdsl-opt %s | mlir-opt --mlir-print-local-scope -mlir-print-op-generic | xdsl-opt

"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<256xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %1 = "memref.alloc"() {alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<1x256xf32>
    "linalg.generic"(%0, %1) ({
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    }) {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>} : (f32, memref<1x256xf32>) -> ()
    "linalg.generic"(%arg0, %arg1, %1) ({
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %3 = "arith.mulf"(%arg3, %arg4) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      %4 = "arith.addf"(%arg5, %3) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%4) : (f32) -> ()
    }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>} : (memref<1x128xf32>, memref<128x256xf32>, memref<1x256xf32>) -> ()
    %2 = "memref.alloc"() {alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<1x256xf32>
    "linalg.generic"(%1, %arg2, %2) ({
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %3 = "arith.addf"(%arg3, %arg4) {fastmath = #arith.fastmath<none>} : (f32, f32) -> f32
      "linalg.yield"(%3) : (f32) -> ()
    }) {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>} : (memref<1x256xf32>, memref<256xf32>, memref<1x256xf32>) -> ()
    "linalg.generic"(%2, %1) ({
    ^bb0(%arg3: f32, %arg4: f32):
      %3 = "arith.cmpf"(%arg3, %0) {predicate = 9 : i64} : (f32, f32) -> i1
      %4 = "arith.select"(%3, %arg3, %0) : (i1, f32, f32) -> f32
      "linalg.yield"(%4) : (f32) -> ()
    }) {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>} : (memref<1x256xf32>, memref<1x256xf32>) -> ()
    "memref.dealloc"(%2) : (memref<1x256xf32>) -> ()
    "func.return"(%1) : (memref<1x256xf32>) -> ()
  }) {function_type = (memref<1x128xf32>, memref<128x256xf32>, memref<256xf32>) -> memref<1x256xf32>, sym_name = "foo"} : () -> ()
}) : () -> ()
