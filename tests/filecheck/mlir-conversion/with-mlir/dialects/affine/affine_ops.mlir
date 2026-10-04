// RUN: xdsl-opt %s | $XDSL_MLIR_OPT --mlir-print-op-generic --mlir-print-local-scope --allow-unregistered-dialect | xdsl-opt | filecheck %s

"builtin.module"() ({

    // For without value being passed during iterations

    affine.for %i = 0 to 256 {
    } {foo = 1 : i32}

    // CHECK:      affine.for %{{.*}} = 0 to 256 {
    // CHECK-NEXT: } {foo = 1 : i32}


    // For with values being passed during iterations

    %init_value = "test.op"() : () -> i32
    %res = affine.for %i = -10 to 10 iter_args(%step_value = %init_value) -> (i32) {
      %next_value = "test.op"() : () -> i32
      affine.yield %next_value : i32
    }
    %00 = "test.op"() : () -> index
    %N = "test.op"() : () -> index
    %res2 = affine.for %i = affine_map<(d0) -> (d0)>(%00) to %N iter_args(%step_value = %init_value) -> (i32) {
      %next_value = "test.op"() : () -> i32
      affine.yield %next_value : i32
    }
    "affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
    ^bb1(%i: index):
      affine.yield
    }) : (index) -> ()

    // CHECK:      %{{.*}} = affine.for %{{.*}} = -10 to 10 iter_args(%{{.*}} = %{{.*}}) -> (i32) {
    // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
    // CHECK-NEXT:   affine.yield %{{.*}} : i32
    // CHECK-NEXT: }
    // CHECK:      %{{.*}} = affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
    // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> i32
    // CHECK-NEXT:   affine.yield %{{.*}} : i32
    // CHECK-NEXT: }
    // CHECK:      "affine.parallel"(%{{.*}}) <{lowerBoundsGroups = dense<1> : vector<1xi32>, lowerBoundsMap = affine_map<() -> (0)>, reductions = [], steps = [1 : i64], upperBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = affine_map<()[s0] -> (s0)>}> ({
    // CHECK-NEXT: ^{{.*}}(%{{.*}}: index):
    // CHECK-NEXT:   affine.yield
    // CHECK-NEXT: }) : (index) -> ()


    %memref = "test.op"() : () -> memref<2x3xf64>
    %value = "test.op"() : () -> f64
    "affine.store"(%value, %memref) <{"map" = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()

    // CHECK:      %{{.*}} = "test.op"() : () -> memref<2x3xf64>
    // CHECK-NEXT: %{{.*}} = "test.op"() : () -> f64
    // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[0, 0] : memref<2x3xf64>

    %zero = "test.op"() : () -> index
    %2 = affine.apply affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))> (%zero)[%zero]
    %min = "affine.min"(%zero) <{"map" = affine_map<(d0) -> ((d0 + 41), d0)>}> : (index) -> index
    %same_value = "affine.load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64

    // CHECK:      %{{.*}} = "test.op"() : () -> index
    // CHECK-NEXT: %{{.*}} = affine.apply affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))> (%{{.*}})[%{{.*}}]
    // CHECK-NEXT: %{{.*}} = affine.min affine_map<(d0) -> ((d0 + 41), d0)> (%{{.*}})
    // CHECK-NEXT: %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x3xf64>

    %vmemref = "test.op"() : () -> memref<2x3xf64>
    %vvalue = "test.op"() : () -> vector<2xf64>
    %one = "test.op"() : () -> index
    affine.vector_store %vvalue, %vmemref[0, 0] : memref<2x3xf64>, vector<2xf64>
    %vloaded = affine.vector_load %vmemref[0, 0] : memref<2x3xf64>, vector<2xf64>
    %vnested = affine.vector_load %vmemref[%zero + 3, %zero * 2 + %one * 5] : memref<2x3xf64>, vector<2xf64>

    // CHECK:      %{{.*}} = "test.op"() : () -> memref<2x3xf64>
    // CHECK-NEXT: %{{.*}} = "test.op"() : () -> vector<2xf64>
    // CHECK-NEXT: %{{.*}} = "test.op"() : () -> index
    // CHECK-NEXT: affine.vector_store %{{.*}}, %{{.*}}[0, 0] : memref<2x3xf64>, vector<2xf64>
    // CHECK-NEXT: %{{.*}} = affine.vector_load %{{.*}}[0, 0] : memref<2x3xf64>, vector<2xf64>
    // CHECK-NEXT: %{{.*}} = affine.vector_load %{{.*}}[%{{.*}} + 3, %{{.*}} * 2 + %{{.*}} * 5] : memref<2x3xf64>, vector<2xf64>

    func.func @empty() {
    affine.for %arg0 = 0 to 10 {
    }
    "affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
      affine.yield
    }, {
    })  : () -> ()
    "affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
      affine.yield
    }, {
      affine.yield
    })  : () -> ()

    func.return
  }
// CHECK:    func.func @empty() {
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:      }
// CHECK-NEXT:      "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
// CHECK-NEXT:        affine.yield
// CHECK-NEXT:      }, {
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
// CHECK-NEXT:        affine.yield
// CHECK-NEXT:      }, {
// CHECK-NEXT:        affine.yield
// CHECK-NEXT:      }) : () -> ()

// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
  func.func @affine_if() -> f32 {
    %0 = arith.constant 0.000000e+00 : f32
    %1 = "affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
      affine.yield %0 : f32
    }, {
      affine.yield %0 : f32
    }) : () -> f32
    func.return %1 : f32
  }
// CHECK:    func.func @affine_if() -> f32 {
// CHECK-NEXT:      %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %{{.*}} = "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
// CHECK-NEXT:        affine.yield %{{.*}} : f32
// CHECK-NEXT:      }, {
// CHECK-NEXT:        affine.yield %{{.*}} : f32
// CHECK-NEXT:      }) : () -> f32
// CHECK-NEXT:      func.return %{{.*}} : f32
// CHECK-NEXT:    }


  // Check that an affine.apply with an affine map is printed correctly.

  %c0 = arith.constant 2 : index
  %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%c0]
  // CHECK: %{{.*}} = arith.constant 2 : index
  // CHECK-NEXT: %{{.*}} = affine.apply affine_map<()[{{.*}}] -> (({{.*}} * 4))> ()[%{{.*}}]

  // For with a non-default step

  affine.for %i3 = 0 to 10 step 2 {
  }

  // CHECK: affine.for %{{.*}} = 0 to 10 step 2 {
  // CHECK-NEXT: }

  // For with a single loop-carried value's type given without parentheses

  %bare_init = "test.op"() : () -> index
  %bare_res = affine.for %i4 = 0 to 10 iter_args(%bare_iv = %bare_init) -> index {
    affine.yield %bare_iv : index
  }

  // CHECK: %{{.*}} = affine.for %{{.*}} = 0 to 10 iter_args(%{{.*}} = %{{.*}}) -> (index) {
  // CHECK-NEXT:   affine.yield %{{.*}} : index
  // CHECK-NEXT: }

  // For with a multi-result bound requiring a `max`/`min` prefix, and a bound
  // map with both dimension and symbol operands

  %bound_d = "test.op"() : () -> index
  %bound_s = "test.op"() : () -> index
  affine.for %i5 = max affine_map<(d0)[s0] -> (d0, s0)>(%bound_d)[%bound_s] to 10 {
  }

  // CHECK:      affine.for %{{.*}} = max affine_map<(d0)[s0] -> (d0, s0)>(%{{.*}})[%{{.*}}] to 10 {
  // CHECK-NEXT: }

  // For with a `min` prefix on the upper bound

  affine.for %i6 = max affine_map<(d0)[s0] -> (d0, s0)>(%bound_d)[%bound_s] to min affine_map<(d0) -> (d0, 10)>(%bound_d) {
  }

  // CHECK:      affine.for %{{.*}} = max affine_map<(d0)[s0] -> (d0, s0)>(%{{.*}})[%{{.*}}] to min affine_map<(d0) -> (d0, 10)>(%{{.*}}) {
  // CHECK-NEXT: }

  // For with a plain (non-min/max) affine map as the upper bound

  %ub_d = "test.op"() : () -> index
  affine.for %i7 = 0 to affine_map<(d0) -> (4 * d0)>(%ub_d) {
  }

  // CHECK:      %{{.*}} = "test.op"() : () -> index
  // CHECK-NEXT: affine.for %{{.*}} = 0 to affine_map<(d0) -> ((d0 * 4))>(%{{.*}}) {
  // CHECK-NEXT: }

  // For with multiple loop-carried values

  %ia_init0 = "test.op"() : () -> index
  %ia_init1 = "test.op"() : () -> index
  %ia_res0, %ia_res1 = affine.for %i8 = 0 to 10 iter_args(%a = %ia_init0, %b = %ia_init1) -> (index, index) {
    affine.yield %a, %b : index, index
  }

  // CHECK:      %{{.*}} = "test.op"() : () -> index
  // CHECK-NEXT: %{{.*}} = "test.op"() : () -> index
  // CHECK-NEXT: %{{.*}}, %{{.*}} = affine.for %{{.*}} = 0 to 10 iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (index, index) {
  // CHECK-NEXT:   affine.yield %{{.*}}, %{{.*}} : index, index
  // CHECK-NEXT: }

}) : () -> ()
