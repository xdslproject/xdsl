// RUN: xdsl-opt "%s" -p lower-affine --allow-unregistered-dialect --print-op-generic | filecheck "%s"

"builtin.module"() ({
  %v0, %m = "test.op"() : () -> (f32, memref<2x3xf32>)
  "affine.store"(%v0, %m) {"map" = affine_map<() -> (1, 2)>} : (f32, memref<2x3xf32>) -> ()
  %v1 = "affine.load"(%m) {"map" = affine_map<() -> (1, 2)>} : (memref<2x3xf32>) -> f32
  %v2 = "affine.for"(%v1) <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (2)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 1>}> ({
  ^0(%r : index, %acc0 : f32):
    %v3 = "affine.for"(%acc0) <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (3)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 1>}> ({
    ^2(%c : index, %acc1 : f32):
      %v4 = "affine.load"(%m, %r, %c) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf32>, index, index) -> f32
      %acc_new = "test.op"(%acc1, %v4) : (f32, f32) -> f32
      "affine.yield"(%acc_new) : (f32) -> ()
    }) : (f32) -> f32
    "affine.yield"(%v3) : (f32) -> ()
  }) : (f32) -> f32
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:     %v0, %m = "test.op"() : () -> (f32, memref<2x3xf32>)
// CHECK-NEXT:     %0 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "arith.constant"() <{"value" = 2 : index}> : () -> index
// CHECK-NEXT:     "memref.store"(%v0, %m, %0, %1) : (f32, memref<2x3xf32>, index, index) -> ()
// CHECK-NEXT:     %2 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:     %3 = "arith.constant"() <{"value" = 2 : index}> : () -> index
// CHECK-NEXT:     %v1 = "memref.load"(%m, %2, %3) : (memref<2x3xf32>, index, index) -> f32
// CHECK-NEXT:     %4 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:     %5 = "arith.constant"() <{"value" = 2 : index}> : () -> index
// CHECK-NEXT:     %6 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:     %v2 = "scf.for"(%4, %5, %6, %v1) ({
// CHECK-NEXT:     ^0(%r : index, %acc0 : f32):
// CHECK-NEXT:       %7 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT:       %8 = "arith.constant"() <{"value" = 3 : index}> : () -> index
// CHECK-NEXT:       %9 = "arith.constant"() <{"value" = 1 : index}> : () -> index
// CHECK-NEXT:       %v3 = "scf.for"(%7, %8, %9, %acc0) ({
// CHECK-NEXT:       ^1(%c : index, %acc1 : f32):
// CHECK-NEXT:         %v4 = "memref.load"(%m, %r, %c) : (memref<2x3xf32>, index, index) -> f32
// CHECK-NEXT:         %acc_new = "test.op"(%acc1, %v4) : (f32, f32) -> f32
// CHECK-NEXT:         "scf.yield"(%acc_new) : (f32) -> ()
// CHECK-NEXT:       }) : (index, index, index, f32) -> f32
// CHECK-NEXT:       "scf.yield"(%v3) : (f32) -> ()
// CHECK-NEXT:     }) : (index, index, index, f32) -> f32
// CHECK-NEXT:   }) : () -> ()


// CHECK-NOT: "affine.
