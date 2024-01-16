// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s

"builtin.module"() ({

    // For without value being passed during iterations

    "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (256)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 0>}> ({
    ^0(%i : index):
      "affine.yield"() : () -> ()
    }) : () -> ()

    // CHECK:      "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (256)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 0>}> ({
    // CHECK-NEXT: ^0(%{{.*}} : index):
    // CHECK-NEXT:   "affine.yield"() : () -> ()
    // CHECK-NEXT: }) : () -> ()


    // For with values being passed during iterations

    %init_value = "test.op"() : () -> i32
    %res = "affine.for"(%init_value) <{"lowerBoundMap" = affine_map<() -> (-10)>, "upperBoundMap" = affine_map<() -> (10)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 1>}> ({
    ^1(%i : index, %step_value : i32):
      %next_value = "test.op"() : () -> i32
      "affine.yield"(%next_value) : (i32) -> ()
    }) : (i32) -> (i32)
    %00 = "test.op"() : () -> index
    %N = "test.op"() : () -> index
    %res2 = "affine.for"(%00, %N, %init_value) <{"lowerBoundMap" = affine_map<(d0) -> (d0)>, "upperBoundMap" = affine_map<()[s0] -> (s0)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 1, 1, 1>}> ({
    ^1(%i : index, %step_value : i32):
      %next_value = "test.op"() : () -> i32
      "affine.yield"(%next_value) : (i32) -> ()
    }) : (index, index, i32) -> (i32)

    // CHECK:      %res = "affine.for"(%{{.*}}) <{"lowerBoundMap" = affine_map<() -> (-10)>, "upperBoundMap" = affine_map<() -> (10)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 1>}> ({
    // CHECK-NEXT: ^1(%{{.*}} : index, %{{.*}} : i32):
    // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
    // CHECK-NEXT:   "affine.yield"(%{{.*}}) : (i32) -> ()
    // CHECK-NEXT: }) : (i32) -> i32


    %memref = "test.op"() : () -> memref<2x3xf64>
    %value = "test.op"() : () -> f64
    "affine.store"(%value, %memref) <{"map" = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()

    // CHECK:      %memref = "test.op"() : () -> memref<2x3xf64>
    // CHECK-NEXT: %value = "test.op"() : () -> f64
    // CHECK-NEXT: "affine.store"(%value, %memref) <{"map" = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()

    %zero = "test.op"() : () -> index
    %2 = "affine.apply"(%zero, %zero) <{"map" = affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))>}> : (index, index) -> index
    %min = "affine.min"(%zero) <{"map" = affine_map<(d0) -> ((d0 + 41), d0)>}> : (index) -> index
    %same_value = "affine.load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64

    // CHECK:      %zero = "test.op"() : () -> index
    // CHECK-NEXT: %{{.*}} = "affine.apply"(%{{.*}}, %{{.*}}) <{"map" = affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))>}> : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = "affine.min"(%{{.*}}) <{"map" = affine_map<(d0) -> ((d0 + 41), d0)>}> : (index) -> index
    // CHECK-NEXT: %same_value = "affine.load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64

    func.func @empty() {
    "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "step" = 1 : index, "upperBoundMap" = affine_map<() -> (10)>, "operandSegmentSizes" = array<i32: 0, 0, 0>}> ({
    ^2(%arg0 : index):
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.if"() ({
      "affine.yield"() : () -> ()
    }, {
    }) {"condition" = affine_set<() : (0 == 0)>} : () -> ()
    "affine.if"() ({
      "affine.yield"() : () -> ()
    }, {
      "affine.yield"() : () -> ()
    }) {"condition" = affine_set<() : (0 == 0)>} : () -> ()
    
    func.return
  }
// CHECK:    func.func @empty() {
// CHECK-NEXT:      "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "step" = 1 : index, "upperBoundMap" = affine_map<() -> (10)>, "operandSegmentSizes" = array<i32: 0, 0, 0>}> ({
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }, {
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) : () -> ()

// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
  func.func @affine_if() -> f32 {
    %0 = arith.constant 0.000000e+00 : f32
    %1 = "affine.if"() ({
      "affine.yield"(%0) : (f32) -> ()
    }, {
      "affine.yield"(%0) : (f32) -> ()
    }) {"condition" = affine_set<() : (0 == 0)>} : () -> f32
    func.return %1 : f32
  }
// CHECK:    func.func @affine_if() -> f32 {
// CHECK-NEXT:      %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %{{.*}} = "affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
// CHECK-NEXT:        "affine.yield"(%{{.*}}) : (f32) -> ()
// CHECK-NEXT:      }, {
// CHECK-NEXT:        "affine.yield"(%{{.*}}) : (f32) -> ()
// CHECK-NEXT:      }) : () -> f32
// CHECK-NEXT:      func.return %{{.*}} : f32
// CHECK-NEXT:    }

}) : () -> ()
