// RUN: xdsl-opt -p expand-math-to-polynomials %s | filecheck %s
// RUN: xdsl-opt -p expand-math-to-polynomials{terms=5} %s | filecheck %s --check-prefix=CHECK-FIVE

builtin.module {
  func.func @test_f64(%x: f64) -> f64 {
    %y = math.exp %x : f64
    func.return %y : f64
  }
  func.func @test_f64_terms3(%x: f64) -> f64 {
    %y = math.exp %x {terms = 3 : i64} : f64
    func.return %y : f64
  }
  func.func @test_f32(%x: f32) -> f32 {
    %y = math.exp %x : f32
    func.return %y : f32
  }
  func.func @test_f16(%x: f16) -> f16 {
    %y = math.exp %x : f16
    func.return %y : f16
  }
  func.func @test_vec(%x: vector<2xf32>) -> vector<2xf32> {
    %y = math.exp %x : vector<2xf32>
    func.return %y : vector<2xf32>
  }
}

// CHECK: builtin.module {

// ----- f64 without terms: not expanded -----

// CHECK:      func.func @test_f64(%{{.*}}: f64) -> f64 {
// CHECK-NEXT:   %{{.*}} = math.exp %{{.*}} : f64
// CHECK-NEXT:   func.return %{{.*}} : f64
// CHECK-NEXT: }

// ----- f64 with terms=3 attribute: expanded (2 loop iterations) -----

// CHECK:      func.func @test_f64_terms3(%[[X:.*]]: f64) -> f64 {
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
// CHECK-NEXT:   func.return %[[RES2]] : f64
// CHECK-NEXT: }

// ----- f32 without terms: not expanded -----

// CHECK:      func.func @test_f32(%{{.*}}: f32) -> f32 {
// CHECK-NEXT:   %{{.*}} = math.exp %{{.*}} : f32
// CHECK-NEXT:   func.return %{{.*}} : f32
// CHECK-NEXT: }

// ----- f16 without terms: not expanded -----

// CHECK:      func.func @test_f16(%{{.*}}: f16) -> f16 {
// CHECK-NEXT:   %{{.*}} = math.exp %{{.*}} : f16
// CHECK-NEXT:   func.return %{{.*}} : f16
// CHECK-NEXT: }

// ----- vector<2xf32> without terms: not expanded -----

// CHECK:      func.func @test_vec(%{{.*}}: vector<2xf32>) -> vector<2xf32> {
// CHECK-NEXT:   %{{.*}} = math.exp %{{.*}} : vector<2xf32>
// CHECK-NEXT:   func.return %{{.*}} : vector<2xf32>
// CHECK-NEXT: }

// CHECK: }

// ===== With terms=5 pass parameter: all expanded (4 loop iterations) =====

// CHECK-FIVE: builtin.module {

// ----- f64 expanded with 5 terms (from pass parameter) -----

// CHECK-FIVE:      func.func @test_f64(%[[X:.*]]: f64) -> f64 {
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

// ----- f64 with terms=3 attribute: expanded with 3 terms (op attribute takes priority) -----

// CHECK-FIVE:      func.func @test_f64_terms3(%[[X:.*]]: f64) -> f64 {
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
// CHECK-FIVE-NEXT:   func.return %[[RES2]] : f64
// CHECK-FIVE-NEXT: }

// ----- f32 expanded with 5 terms (from pass parameter) -----

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

// ----- f16 expanded with 5 terms (from pass parameter) -----

// CHECK-FIVE:      func.func @test_f16(%[[X16:.*]]: f16) -> f16 {
// CHECK-FIVE-NEXT:   %[[RES0_16:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-FIVE-NEXT:   %[[TERM0_16:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-FIVE-NEXT:   %[[I1_16:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-FIVE-NEXT:   %[[FRAC1_16:.*]] = arith.mulf %[[X16]], %[[I1_16]] : f16
// CHECK-FIVE-NEXT:   %[[TERM1_16:.*]] = arith.mulf %[[FRAC1_16]], %[[TERM0_16]] : f16
// CHECK-FIVE-NEXT:   %[[RES1_16:.*]] = arith.addf %[[RES0_16]], %[[TERM1_16]] : f16
// CHECK-FIVE-NEXT:   %[[I2_16:.*]] = arith.constant 5.000000e-01 : f16
// CHECK-FIVE-NEXT:   %[[FRAC2_16:.*]] = arith.mulf %[[X16]], %[[I2_16]] : f16
// CHECK-FIVE-NEXT:   %[[TERM2_16:.*]] = arith.mulf %[[FRAC2_16]], %[[TERM1_16]] : f16
// CHECK-FIVE-NEXT:   %[[RES2_16:.*]] = arith.addf %[[RES1_16]], %[[TERM2_16]] : f16
// CHECK-FIVE-NEXT:   %[[I3_16:.*]] = arith.constant 3.332520e-01 : f16
// CHECK-FIVE-NEXT:   %[[FRAC3_16:.*]] = arith.mulf %[[X16]], %[[I3_16]] : f16
// CHECK-FIVE-NEXT:   %[[TERM3_16:.*]] = arith.mulf %[[FRAC3_16]], %[[TERM2_16]] : f16
// CHECK-FIVE-NEXT:   %[[RES3_16:.*]] = arith.addf %[[RES2_16]], %[[TERM3_16]] : f16
// CHECK-FIVE-NEXT:   %[[I4_16:.*]] = arith.constant 2.500000e-01 : f16
// CHECK-FIVE-NEXT:   %[[FRAC4_16:.*]] = arith.mulf %[[X16]], %[[I4_16]] : f16
// CHECK-FIVE-NEXT:   %[[TERM4_16:.*]] = arith.mulf %[[FRAC4_16]], %[[TERM3_16]] : f16
// CHECK-FIVE-NEXT:   %[[RES4_16:.*]] = arith.addf %[[RES3_16]], %[[TERM4_16]] : f16
// CHECK-FIVE-NEXT:   func.return %[[RES4_16]] : f16
// CHECK-FIVE-NEXT: }

// ----- vector<2xf32> expanded with 5 terms (from pass parameter) -----

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
