// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({

    // For without value being passed during iterations

    "affine.for"() <{"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (256)>, "step" = 1 : index}> ({
    ^0(%i : index):
      "affine.yield"() : () -> ()
    }) : () -> ()

    // CHECK:      "affine.for"() <{"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (256)>, "step" = 1 : index}> ({
    // CHECK-NEXT: ^0(%{{.*}} : index):
    // CHECK-NEXT:   "affine.yield"() : () -> ()
    // CHECK-NEXT: }) : () -> ()


    // For with values being passed during iterations

    %init_value = "test.op"() : () -> !test.type<"int">
    %res = "affine.for"(%init_value) <{"lower_bound" = affine_map<() -> (-10)>, "upper_bound" = affine_map<() -> (10)>, "step" = 1 : index}> ({
    ^1(%i : index, %step_value : !test.type<"int">):
      %next_value = "test.op"() : () -> !test.type<"int">
      "affine.yield"(%next_value) : (!test.type<"int">) -> ()
    }) : (!test.type<"int">) -> (!test.type<"int">)
    %00 = "test.op"() : () -> index
    %N = "test.op"() : () -> index
    %res2 = "affine.for"(%00, %N, %init_value) <{"lower_bound" = affine_map<(d0) -> (d0)>, "upper_bound" = affine_map<()[s0] -> (s0)>, "step" = 1 : index}> ({
    ^1(%i : index, %step_value : !test.type<"int">):
      %next_value = "test.op"() : () -> !test.type<"int">
      "affine.yield"(%next_value) : (!test.type<"int">) -> ()
    }) : (index, index, !test.type<"int">) -> (!test.type<"int">)

    // CHECK:      %res = "affine.for"(%{{.*}}) <{"lower_bound" = affine_map<() -> (-10)>, "upper_bound" = affine_map<() -> (10)>, "step" = 1 : index}> ({
    // CHECK-NEXT: ^1(%{{.*}} : index, %{{.*}} : !test.type<"int">):
    // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> !test.type<"int">
    // CHECK-NEXT:   "affine.yield"(%{{.*}}) : (!test.type<"int">) -> ()
    // CHECK-NEXT: }) : (!test.type<"int">) -> !test.type<"int">


    %memref = "test.op"() : () -> memref<2x3xf64>
    %value = "test.op"() : () -> f64
    "affine.store"(%value, %memref) <{"map" = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()

    // CHECK:      %memref = "test.op"() : () -> memref<2x3xf64>
    // CHECK-NEXT: %value = "test.op"() : () -> f64
    // CHECK-NEXT: "affine.store"(%value, %memref) <{"map" = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()

    %zero = "test.op"() : () -> index
    %2 = "affine.apply"(%zero, %zero) <{"map" = affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))>}> : (index, index) -> index
    %same_value = "affine.load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64

    // CHECK:      %zero = "test.op"() : () -> index
    // CHECK-NEXT: %{{.*}} = "affine.apply"(%{{.*}}, %{{.*}}) <{"map" = affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))>}> : (index, index) -> index
    // CHECK-NEXT: %same_value = "affine.load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64

    func.func @empty() {
    "affine.for"() ({
    ^2(%arg0 : index):
      "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "some_attr" = true, "step" = 1 : index, "upper_bound" = affine_map<() -> (10)>} : () -> ()
    "affine.if"() ({
      "affine.yield"() : () -> ()
    }, {
    }) {"condition" = affine_set<() : (0 == 0)>, "some_attr" = true} : () -> ()
    "affine.if"() ({
      "affine.yield"() : () -> ()
    }, {
      "affine.yield"() : () -> ()
    }) {"condition" = affine_set<() : (0 == 0)>, "some_attr" = true} : () -> ()
    
    func.return
  }
// CHECK:    func.func @empty() {
// CHECK-NEXT:      "affine.for"() <{"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (10)>, "step" = 1 : index}> ({
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) {"some_attr" = true} : () -> ()
// CHECK-NEXT:      "affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) {"some_attr" = true} : () -> ()
// CHECK-NEXT:      "affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }, {
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) {"some_attr" = true} : () -> ()

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
