// RUN: xdsl-opt -p expand-math-to-polynomials %s | filecheck %s
// RUN: xdsl-opt -p expand-math-to-polynomials{terms=5} %s | filecheck %s --check-prefix=CHECK-FIVE

builtin.module {
  func.func @test(%x: f64) -> f64 {
    %y = math.exp %x : f64
    func.return %y : f64
  }
  func.func @test_f32(%x: f32) -> f32 {
    %y = math.exp %x : f32
    func.return %y : f32
  }
  func.func @test_vec(%x: vector<2xf32>) -> vector<2xf32> {
    %y = math.exp %x : vector<2xf32>
    func.return %y : vector<2xf32>
  }
}

// CHECK: builtin.module {

// Anchor @test and capture the argument SSA name
// CHECK: func.func @test(%[[X:.*]] {{:}} f64) -> f64

// In @test, exp must be gone
// CHECK-NOT: math.exp

// Seed constants (these should exist if your pass always inserts them)
// CHECK: %[[RES0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK: %[[TERM0:.*]] = arith.constant 1.000000e+00 : f64

// First iteration shape uses the argument, not a constant X
// CHECK: %[[I1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK: %[[FRAC1:.*]] = arith.divf %[[X]], %[[I1]] : f64
// CHECK: %[[TERM1:.*]] = arith.mulf %[[FRAC1]], %[[TERM0]] : f64
// CHECK: %[[RES1:.*]] = arith.addf %[[RES0]], %[[TERM1]] : f64

// Test counts of Loop iterations (3 in total but we check first iteration already so that divf is consumed previously)
// CHECK-COUNT-2: arith.divf

// Return exists
// CHECK: func.return %{{.*}} : f64

// ----- f32 test -----

// CHECK: func.func @test_f32(%[[X32:.*]] {{:}} f32) -> f32
// CHECK-NOT: math.exp
// CHECK: %[[RES0_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[TERM0_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[I1_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[FRAC1_32:.*]] = arith.divf %[[X32]], %[[I1_32]] : f32
// CHECK: %[[TERM1_32:.*]] = arith.mulf %[[FRAC1_32]], %[[TERM0_32]] : f32
// CHECK: %[[RES1_32:.*]] = arith.addf %[[RES0_32]], %[[TERM1_32]] : f32
// CHECK-COUNT-2: arith.divf
// CHECK: func.return %{{.*}} : f32

// ----- vector<2xf32> test -----

// CHECK: func.func @test_vec(%[[XV:.*]] {{:}} vector<2xf32>) -> vector<2xf32>
// CHECK-NOT: math.exp
// CHECK: %[[RES0_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK: %[[TERM0_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK: %[[I1_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK: %[[FRAC1_V:.*]] = arith.divf %[[XV]], %[[I1_V]] : vector<2xf32>
// CHECK: %[[TERM1_V:.*]] = arith.mulf %[[FRAC1_V]], %[[TERM0_V]] : vector<2xf32>
// CHECK: %[[RES1_V:.*]] = arith.addf %[[RES0_V]], %[[TERM1_V]] : vector<2xf32>
// CHECK-COUNT-2: arith.divf
// CHECK: func.return %{{.*}} : vector<2xf32>

// ----- terms=5 produces 4 loop iterations -----

// CHECK-FIVE: builtin.module {
// CHECK-FIVE: func.func @test(%[[X:.*]] {{:}} f64) -> f64
// CHECK-FIVE-NOT: math.exp

// Seed constants
// CHECK-FIVE: %[[RES0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-FIVE: %[[TERM0:.*]] = arith.constant 1.000000e+00 : f64

// First iteration
// CHECK-FIVE: %[[I1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-FIVE: %[[FRAC1:.*]] = arith.divf %[[X]], %[[I1]] : f64
// CHECK-FIVE: %[[TERM1:.*]] = arith.mulf %[[FRAC1]], %[[TERM0]] : f64
// CHECK-FIVE: %[[RES1:.*]] = arith.addf %[[RES0]], %[[TERM1]] : f64

// Remaining 3 iterations
// CHECK-FIVE-COUNT-3: arith.divf

// CHECK-FIVE: func.return %{{.*}} : f64

// CHECK-FIVE: func.func @test_f32
// CHECK-FIVE-NOT: math.exp
// CHECK-FIVE-COUNT-4: arith.divf
// CHECK-FIVE: func.return %{{.*}} : f32

// CHECK-FIVE: func.func @test_vec
// CHECK-FIVE-NOT: math.exp
// CHECK-FIVE-COUNT-4: arith.divf
// CHECK-FIVE: func.return %{{.*}} : vector<2xf32>
