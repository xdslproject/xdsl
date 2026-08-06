// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({

    // For without value being passed during iterations, and discardable attribute

    affine.for %i2 = 0 to 4 {
    } {foo = 1 : i32}

    // CHECK:      affine.for %{{.*}} = 0 to 4 {
    // CHECK-NEXT: } {foo = 1 : i32}


    // For with values being passed during iterations

    %init_value = "test.op"() : () -> !test.type<"int">
    %res = affine.for %i = -10 to 10 iter_args(%step_value = %init_value) -> (!test.type<"int">) {
      %next_value = "test.op"() : () -> !test.type<"int">
      affine.yield %next_value : !test.type<"int">
    }
    %00 = "test.op"() : () -> index
    %N = "test.op"() : () -> index
    %res2 = affine.for %i = affine_map<(d0) -> (d0)>(%00) to %N iter_args(%step_value = %init_value) -> (!test.type<"int">) {
      %next_value = "test.op"() : () -> !test.type<"int">
      affine.yield %next_value : !test.type<"int">
    }
    "affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
    ^bb1(%i: index):
      affine.yield
    }) : (index) -> ()

    // CHECK:      %res = affine.for %{{.*}} = -10 to 10 iter_args(%{{.*}} = %{{.*}}) -> (!test.type<"int">) {
    // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> !test.type<"int">
    // CHECK-NEXT:   affine.yield %{{.*}} : !test.type<"int">
    // CHECK-NEXT: }
    // CHECK:      %res2 = affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to %N iter_args(%{{.*}} = %{{.*}}) -> (!test.type<"int">) {
    // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> !test.type<"int">
    // CHECK-NEXT:   affine.yield %{{.*}} : !test.type<"int">
    // CHECK-NEXT: }
    // CHECK:      "affine.parallel"(%N) <{lowerBoundsMap = affine_map<() -> (0)>, lowerBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = affine_map<()[s0] -> (s0)>, upperBoundsGroups = dense<1> : vector<1xi32>, steps = [1 : i64], reductions = []}> ({
    // CHECK-NEXT: ^{{.*}}(%{{.*}}: index):
    // CHECK-NEXT:   affine.yield
    // CHECK-NEXT: }) : (index) -> ()


    %memref = "test.op"() : () -> memref<2x3xf64>
    %value = "test.op"() : () -> f64
    affine.store %value, %memref[0, 0] : memref<2x3xf64>

    // CHECK:      %memref = "test.op"() : () -> memref<2x3xf64>
    // CHECK-NEXT: %value = "test.op"() : () -> f64
    // CHECK-NEXT: affine.store %value, %memref[0, 0] : memref<2x3xf64>

    %zero = "test.op"() : () -> index
    %2 = affine.apply affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))> (%zero)[%zero]
    %min = "affine.min"(%zero) <{"map" = affine_map<(d0) -> ((d0 + 41), d0)>}> : (index) -> index
    %same_value = affine.load %memref[%zero, %zero] : memref<2x3xf64>
    %nested = affine.load %memref[3 + %zero * 7 + %zero, %zero + 7] : memref<2x3xf64>

    // CHECK:      %zero = "test.op"() : () -> index
    // CHECK-NEXT: %{{.*}} = affine.apply affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))> (%{{.*}})[%{{.*}}]
    // CHECK-NEXT: %{{.*}} = "affine.min"(%{{.*}}) <{map = affine_map<(d0) -> ((d0 + 41), d0)>}> : (index) -> index
    // CHECK-NEXT: %same_value = affine.load %memref[%zero, %zero] : memref<2x3xf64>
    // CHECK-NEXT: %nested = affine.load %memref[%zero * 7 + 3 + %zero, %zero + 7] : memref<2x3xf64>

    %vmemref = "test.op"() : () -> memref<2x3xf64>
    %vvalue = "test.op"() : () -> vector<2xf64>
    affine.vector_store %vvalue, %vmemref[0, 0] : memref<2x3xf64>, vector<2xf64>
    %vloaded = affine.vector_load %vmemref[0, 0] : memref<2x3xf64>, vector<2xf64>
    %vnested = affine.vector_load %vmemref[%zero + 3, %zero * 2 + %zero * 5] : memref<2x3xf64>, vector<2xf64>

    // CHECK:      %vmemref = "test.op"() : () -> memref<2x3xf64>
    // CHECK-NEXT: %vvalue = "test.op"() : () -> vector<2xf64>
    // CHECK-NEXT: affine.vector_store %vvalue, %vmemref[0, 0] : memref<2x3xf64>, vector<2xf64>
    // CHECK-NEXT: %vloaded = affine.vector_load %vmemref[0, 0] : memref<2x3xf64>, vector<2xf64>
    // CHECK-NEXT: %vnested = affine.vector_load %vmemref[%zero + 3, %zero * 2 + %zero * 5] : memref<2x3xf64>, vector<2xf64>

    func.func @empty() {
    affine.for %arg0 = 0 to 10 {
    }
    "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
      affine.yield
    }, {
    })  : () -> ()
    "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
      affine.yield
    }, {
      affine.yield
    }) : () -> ()

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
    %1 = "affine.if"() <{condition = affine_set<() : (0 == 0)>}> ({
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

  // CHECK: %bare_res = affine.for %{{.*}} = 0 to 10 iter_args(%{{.*}} = %{{.*}}) -> (index) {
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

}) : () -> ()
