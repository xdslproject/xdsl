// RUN: xdsl-opt -p expand-exp-to-polynomials %s | filecheck %s

builtin.module {
  func.func @test(%x: f64) -> f64 {
    %y = math.exp %x : f64
    func.return %y : f64
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

// Test counts of Loop iterations (74 in total but we check first iteration already so that divf is consumed previously)
// CHECK-COUNT-73: arith.divf

// Return exists (don’t try to bind the last SSA without unrolling 74 iters)
// CHECK: func.return %{{.*}} : f64
