// RUN: xdsl-opt %s -p lower-affine | filecheck %s

// CHECK:      builtin.module {

// CHECK-NEXT:    %v0, %m = "test.op"() : () -> (f32, memref<2x3xf32>)
%v0, %m = "test.op"() : () -> (f32, memref<2x3xf32>)

// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    memref.store %v0, %m[%{{.*}}, %{{.*}}] : memref<2x3xf32>
"affine.store"(%v0, %m) <{"map" = affine_map<() -> (1, 2)>}> : (f32, memref<2x3xf32>) -> ()

// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %v1 = memref.load %m[%{{.*}}, %{{.*}}] : memref<2x3xf32>
%v1 = "affine.load"(%m) <{"map" = affine_map<() -> (1, 2)>}> : (memref<2x3xf32>) -> f32

// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    %v2 = scf.for %r = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%acc0 = %v1) -> (f32) {
%v2 = "affine.for"(%v1) <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (2)>, "step" = 1 : index, operandSegmentSizes = array<i32: 0, 0, 1>}> ({
^bb0(%r : index, %acc0 : f32):

// CHECK-NEXT:      %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:      %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:      %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:      %v3 = scf.for %c = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%acc1 = %acc0) -> (f32) {
  %v3 = "affine.for"(%acc0) <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (3)>, "step" = 1 : index, operandSegmentSizes = array<i32: 0, 0, 1>}> ({
  ^bb1(%c : index, %acc1 : f32):

// CHECK-NEXT:        %v4 = memref.load %m[%r, %c] : memref<2x3xf32>
    %v4 = "affine.load"(%m, %r, %c) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf32>, index, index) -> f32

// CHECK-NEXT:        %acc_new = "test.op"(%acc1, %v4) : (f32, f32) -> f32
    %acc_new = "test.op"(%acc1, %v4) : (f32, f32) -> f32

// CHECK-NEXT:        scf.yield %acc_new : f32
    "affine.yield"(%acc_new) : (f32) -> ()

// CHECK-NEXT:      }
  }) : (f32) -> f32

// CHECK-NEXT:      scf.yield %v3 : f32
  "affine.yield"(%v3) : (f32) -> ()

// CHECK-NEXT:    }
}) : (f32) -> f32

// CHECK-NEXT:    %apply_dim, %apply_sym = "test.op"() : () -> (index, index)
// CHECK-NEXT:    %apply_res = arith.constant 42 : index
// CHECK-NEXT:    %apply_res_1 = arith.muli %apply_sym, %apply_res : index
// CHECK-NEXT:    %apply_res_2 = arith.addi %apply_dim, %apply_res_1 : index
// CHECK-NEXT:    %apply_res_3 = arith.constant -1 : index
// CHECK-NEXT:    %apply_res_4 = arith.addi %apply_res_2, %apply_res_3 : index
%apply_dim, %apply_sym = "test.op"() : () -> (index, index)
%apply_res = affine.apply affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))> (%apply_dim)[%apply_sym]

// CHECK-NEXT:  }

// CHECK-NOT: "affine.
