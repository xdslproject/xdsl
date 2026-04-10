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

// ----- f64 test (default terms=4, i.e. 3 loop iterations) -----

// CHECK:      func.func @test(%[[X:.*]]: f64) -> f64 {
// CHECK-NEXT:   %[[RES0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %[[TERM0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %[[I1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %[[FRAC1:.*]] = arith.mulf %[[X]], %[[I1]] : f64
// CHECK-NEXT:   %[[TERM1:.*]] = arith.mulf %[[FRAC1]], %[[TERM0]] : f64
// CHECK-NEXT:   %[[RES1:.*]] = arith.addf %[[RES0]], %[[TERM1]] : f64
// CHECK-NEXT:   %[[I2:.*]] = arith.constant 5.000000e-01 : f64
// CHECK-NEXT:   %[[FRAC2:.*]] = arith.mulf %[[X]], %[[I2]] : f64
// CHECK-NEXT:   %[[TERM2:.*]] = arith.mulf %[[FRAC2]], %[[TERM1]] : f64
// CHECK-NEXT:   %[[RES2:.*]] = arith.addf %[[RES1]], %[[TERM2]] : f64
// CHECK-NEXT:   %[[I3:.*]] = arith.constant 0.33333333333333331 : f64
// CHECK-NEXT:   %[[FRAC3:.*]] = arith.mulf %[[X]], %[[I3]] : f64
// CHECK-NEXT:   %[[TERM3:.*]] = arith.mulf %[[FRAC3]], %[[TERM2]] : f64
// CHECK-NEXT:   %[[RES3:.*]] = arith.addf %[[RES2]], %[[TERM3]] : f64
// CHECK-NEXT:   func.return %[[RES3]] : f64
// CHECK-NEXT: }

// ----- f32 test -----

// CHECK:      func.func @test_f32(%[[X32:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[RES0_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %[[TERM0_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %[[I1_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %[[FRAC1_32:.*]] = arith.mulf %[[X32]], %[[I1_32]] : f32
// CHECK-NEXT:   %[[TERM1_32:.*]] = arith.mulf %[[FRAC1_32]], %[[TERM0_32]] : f32
// CHECK-NEXT:   %[[RES1_32:.*]] = arith.addf %[[RES0_32]], %[[TERM1_32]] : f32
// CHECK-NEXT:   %[[I2_32:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:   %[[FRAC2_32:.*]] = arith.mulf %[[X32]], %[[I2_32]] : f32
// CHECK-NEXT:   %[[TERM2_32:.*]] = arith.mulf %[[FRAC2_32]], %[[TERM1_32]] : f32
// CHECK-NEXT:   %[[RES2_32:.*]] = arith.addf %[[RES1_32]], %[[TERM2_32]] : f32
// CHECK-NEXT:   %[[I3_32:.*]] = arith.constant 0.333333343 : f32
// CHECK-NEXT:   %[[FRAC3_32:.*]] = arith.mulf %[[X32]], %[[I3_32]] : f32
// CHECK-NEXT:   %[[TERM3_32:.*]] = arith.mulf %[[FRAC3_32]], %[[TERM2_32]] : f32
// CHECK-NEXT:   %[[RES3_32:.*]] = arith.addf %[[RES2_32]], %[[TERM3_32]] : f32
// CHECK-NEXT:   func.return %[[RES3_32]] : f32
// CHECK-NEXT: }

// ----- vector<2xf32> test -----

// CHECK:      func.func @test_vec(%[[XV:.*]]: vector<2xf32>) -> vector<2xf32> {
// CHECK-NEXT:   %[[RES0_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK-NEXT:   %[[TERM0_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK-NEXT:   %[[I1_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK-NEXT:   %[[FRAC1_V:.*]] = arith.mulf %[[XV]], %[[I1_V]] : vector<2xf32>
// CHECK-NEXT:   %[[TERM1_V:.*]] = arith.mulf %[[FRAC1_V]], %[[TERM0_V]] : vector<2xf32>
// CHECK-NEXT:   %[[RES1_V:.*]] = arith.addf %[[RES0_V]], %[[TERM1_V]] : vector<2xf32>
// CHECK-NEXT:   %[[I2_V:.*]] = arith.constant dense<5.000000e-01> : vector<2xf32>
// CHECK-NEXT:   %[[FRAC2_V:.*]] = arith.mulf %[[XV]], %[[I2_V]] : vector<2xf32>
// CHECK-NEXT:   %[[TERM2_V:.*]] = arith.mulf %[[FRAC2_V]], %[[TERM1_V]] : vector<2xf32>
// CHECK-NEXT:   %[[RES2_V:.*]] = arith.addf %[[RES1_V]], %[[TERM2_V]] : vector<2xf32>
// CHECK-NEXT:   %[[I3_V:.*]] = arith.constant dense<0.333333343> : vector<2xf32>
// CHECK-NEXT:   %[[FRAC3_V:.*]] = arith.mulf %[[XV]], %[[I3_V]] : vector<2xf32>
// CHECK-NEXT:   %[[TERM3_V:.*]] = arith.mulf %[[FRAC3_V]], %[[TERM2_V]] : vector<2xf32>
// CHECK-NEXT:   %[[RES3_V:.*]] = arith.addf %[[RES2_V]], %[[TERM3_V]] : vector<2xf32>
// CHECK-NEXT:   func.return %[[RES3_V]] : vector<2xf32>
// CHECK-NEXT: }

// ----- terms=5 produces 4 loop iterations -----

// CHECK-FIVE: builtin.module {

// CHECK-FIVE:      func.func @test(%[[X:.*]]: f64) -> f64 {
// CHECK-FIVE-NEXT:   %[[RES0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-FIVE-NEXT:   %[[TERM0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-FIVE-NEXT:   %[[I1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-FIVE-NEXT:   %[[FRAC1:.*]] = arith.mulf %[[X]], %[[I1]] : f64
// CHECK-FIVE-NEXT:   %[[TERM1:.*]] = arith.mulf %[[FRAC1]], %[[TERM0]] : f64
// CHECK-FIVE-NEXT:   %[[RES1:.*]] = arith.addf %[[RES0]], %[[TERM1]] : f64
// CHECK-FIVE-NEXT:   %[[I2:.*]] = arith.constant 5.000000e-01 : f64
// CHECK-FIVE-NEXT:   %[[FRAC2:.*]] = arith.mulf %[[X]], %[[I2]] : f64
// CHECK-FIVE-NEXT:   %[[TERM2:.*]] = arith.mulf %[[FRAC2]], %[[TERM1]] : f64
// CHECK-FIVE-NEXT:   %[[RES2:.*]] = arith.addf %[[RES1]], %[[TERM2]] : f64
// CHECK-FIVE-NEXT:   %[[I3:.*]] = arith.constant 0.33333333333333331 : f64
// CHECK-FIVE-NEXT:   %[[FRAC3:.*]] = arith.mulf %[[X]], %[[I3]] : f64
// CHECK-FIVE-NEXT:   %[[TERM3:.*]] = arith.mulf %[[FRAC3]], %[[TERM2]] : f64
// CHECK-FIVE-NEXT:   %[[RES3:.*]] = arith.addf %[[RES2]], %[[TERM3]] : f64
// CHECK-FIVE-NEXT:   %[[I4:.*]] = arith.constant 2.500000e-01 : f64
// CHECK-FIVE-NEXT:   %[[FRAC4:.*]] = arith.mulf %[[X]], %[[I4]] : f64
// CHECK-FIVE-NEXT:   %[[TERM4:.*]] = arith.mulf %[[FRAC4]], %[[TERM3]] : f64
// CHECK-FIVE-NEXT:   %[[RES4:.*]] = arith.addf %[[RES3]], %[[TERM4]] : f64
// CHECK-FIVE-NEXT:   func.return %[[RES4]] : f64
// CHECK-FIVE-NEXT: }

// CHECK-FIVE:      func.func @test_f32(%[[X32:.*]]: f32) -> f32 {
// CHECK-FIVE-NEXT:   %[[RES0_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-FIVE-NEXT:   %[[TERM0_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-FIVE-NEXT:   %[[I1_32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-FIVE-NEXT:   %[[FRAC1_32:.*]] = arith.mulf %[[X32]], %[[I1_32]] : f32
// CHECK-FIVE-NEXT:   %[[TERM1_32:.*]] = arith.mulf %[[FRAC1_32]], %[[TERM0_32]] : f32
// CHECK-FIVE-NEXT:   %[[RES1_32:.*]] = arith.addf %[[RES0_32]], %[[TERM1_32]] : f32
// CHECK-FIVE-NEXT:   %[[I2_32:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-FIVE-NEXT:   %[[FRAC2_32:.*]] = arith.mulf %[[X32]], %[[I2_32]] : f32
// CHECK-FIVE-NEXT:   %[[TERM2_32:.*]] = arith.mulf %[[FRAC2_32]], %[[TERM1_32]] : f32
// CHECK-FIVE-NEXT:   %[[RES2_32:.*]] = arith.addf %[[RES1_32]], %[[TERM2_32]] : f32
// CHECK-FIVE-NEXT:   %[[I3_32:.*]] = arith.constant 0.333333343 : f32
// CHECK-FIVE-NEXT:   %[[FRAC3_32:.*]] = arith.mulf %[[X32]], %[[I3_32]] : f32
// CHECK-FIVE-NEXT:   %[[TERM3_32:.*]] = arith.mulf %[[FRAC3_32]], %[[TERM2_32]] : f32
// CHECK-FIVE-NEXT:   %[[RES3_32:.*]] = arith.addf %[[RES2_32]], %[[TERM3_32]] : f32
// CHECK-FIVE-NEXT:   %[[I4_32:.*]] = arith.constant 2.500000e-01 : f32
// CHECK-FIVE-NEXT:   %[[FRAC4_32:.*]] = arith.mulf %[[X32]], %[[I4_32]] : f32
// CHECK-FIVE-NEXT:   %[[TERM4_32:.*]] = arith.mulf %[[FRAC4_32]], %[[TERM3_32]] : f32
// CHECK-FIVE-NEXT:   %[[RES4_32:.*]] = arith.addf %[[RES3_32]], %[[TERM4_32]] : f32
// CHECK-FIVE-NEXT:   func.return %[[RES4_32]] : f32
// CHECK-FIVE-NEXT: }

// CHECK-FIVE:      func.func @test_vec(%[[XV:.*]]: vector<2xf32>) -> vector<2xf32> {
// CHECK-FIVE-NEXT:   %[[RES0_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[TERM0_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[I1_V:.*]] = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[FRAC1_V:.*]] = arith.mulf %[[XV]], %[[I1_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[TERM1_V:.*]] = arith.mulf %[[FRAC1_V]], %[[TERM0_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[RES1_V:.*]] = arith.addf %[[RES0_V]], %[[TERM1_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[I2_V:.*]] = arith.constant dense<5.000000e-01> : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[FRAC2_V:.*]] = arith.mulf %[[XV]], %[[I2_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[TERM2_V:.*]] = arith.mulf %[[FRAC2_V]], %[[TERM1_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[RES2_V:.*]] = arith.addf %[[RES1_V]], %[[TERM2_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[I3_V:.*]] = arith.constant dense<0.333333343> : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[FRAC3_V:.*]] = arith.mulf %[[XV]], %[[I3_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[TERM3_V:.*]] = arith.mulf %[[FRAC3_V]], %[[TERM2_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[RES3_V:.*]] = arith.addf %[[RES2_V]], %[[TERM3_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[I4_V:.*]] = arith.constant dense<2.500000e-01> : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[FRAC4_V:.*]] = arith.mulf %[[XV]], %[[I4_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[TERM4_V:.*]] = arith.mulf %[[FRAC4_V]], %[[TERM3_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   %[[RES4_V:.*]] = arith.addf %[[RES3_V]], %[[TERM4_V]] : vector<2xf32>
// CHECK-FIVE-NEXT:   func.return %[[RES4_V]] : vector<2xf32>
// CHECK-FIVE-NEXT: }
