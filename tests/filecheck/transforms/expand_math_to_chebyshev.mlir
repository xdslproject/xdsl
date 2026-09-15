// RUN: xdsl-opt -p expand-math-to-chebyshev %s | filecheck %s
// RUN: xdsl-opt -p 'expand-math-to-chebyshev{degree=4 lower=-5.0 upper=0.0}' %s | filecheck %s --check-prefix=CHECK-PASS

builtin.module {
  func.func @test_exp_f64(%x: f64) -> f64 {
    %y = math.exp %x : f64
    func.return %y : f64
  }
  func.func @test_exp_f64_attrs(%x: f64) -> f64 {
    %y = math.exp %x {degree = 4 : i64, lower = -5.0 : f64, upper = 0.0 : f64} : f64
    func.return %y : f64
  }
  func.func @test_exp_f32(%x: f32) -> f32 {
    %y = math.exp %x : f32
    func.return %y : f32
  }
  func.func @test_exp_f16(%x: f16) -> f16 {
    %y = math.exp %x : f16
    func.return %y : f16
  }
  func.func @test_exp_vec(%x: vector<2xf32>) -> vector<2xf32> {
    %y = math.exp %x : vector<2xf32>
    func.return %y : vector<2xf32>
  }
}

// CHECK: builtin.module {

// ----- f64 without attributes: not expanded -----

// CHECK:      func.func @test_exp_f64(%{{.*}}: f64) -> f64 {
// CHECK-NEXT:   %{{.*}} = math.exp %{{.*}} : f64
// CHECK-NEXT:   func.return %{{.*}} : f64
// CHECK-NEXT: }

// ----- f64 with degree=4 attribute: expanded (Chebyshev on [-5, 0]) -----

// CHECK:      func.func @test_exp_f64_attrs(%[[X:.*]]: f64) -> f64 {
//               Domain mapping: t = x * 0.4 + 1.0
// CHECK-NEXT:   %[[SCALE:.*]] = arith.constant 4.000000e-01 : f64
// CHECK-NEXT:   %[[OFFSET:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %[[SCALED:.*]] = arith.mulf %[[X]], %[[SCALE]] : f64
// CHECK-NEXT:   %[[T:.*]] = arith.addf %[[SCALED]], %[[OFFSET]] : f64
//               2*t
// CHECK-NEXT:   %[[TWO:.*]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %[[TWO_T:.*]] = arith.mulf %[[TWO]], %[[T]] : f64
//               b_{n+2} = 0, b_{n+1} = 0
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
//               4 Clenshaw iterations
// CHECK-COUNT-4: arith.addf
//               Final: result = c_0/2 + t * b_1 - b_2
// CHECK:        %[[C0_HALF:.*]] = arith.constant 0.27007526935428111 : f64
// CHECK-NEXT:   %[[T_B1:.*]] = arith.mulf %[[T]], %{{.*}} : f64
// CHECK-NEXT:   %[[ADD:.*]] = arith.addf %[[C0_HALF]], %[[T_B1]] : f64
// CHECK-NEXT:   %[[RES:.*]] = arith.subf %[[ADD]], %{{.*}} : f64
// CHECK-NEXT:   func.return %[[RES]] : f64
// CHECK-NEXT: }

// ----- f32 without attributes: not expanded -----

// CHECK:      func.func @test_exp_f32(%{{.*}}: f32) -> f32 {
// CHECK-NEXT:   %{{.*}} = math.exp %{{.*}} : f32
// CHECK-NEXT:   func.return %{{.*}} : f32
// CHECK-NEXT: }

// ----- f16 without attributes: not expanded -----

// CHECK:      func.func @test_exp_f16(%{{.*}}: f16) -> f16 {
// CHECK-NEXT:   %{{.*}} = math.exp %{{.*}} : f16
// CHECK-NEXT:   func.return %{{.*}} : f16
// CHECK-NEXT: }

// ----- vector<2xf32> without attributes: not expanded -----

// CHECK:      func.func @test_exp_vec(%{{.*}}: vector<2xf32>) -> vector<2xf32> {
// CHECK-NEXT:   %{{.*}} = math.exp %{{.*}} : vector<2xf32>
// CHECK-NEXT:   func.return %{{.*}} : vector<2xf32>
// CHECK-NEXT: }

// CHECK: }

// ===== With pass parameters: all expanded (degree=4, [-5, 0]) =====

// CHECK-PASS: builtin.module {

// ----- f64 expanded via pass parameter -----

// CHECK-PASS:      func.func @test_exp_f64(%[[X:.*]]: f64) -> f64 {
// CHECK-PASS-NEXT:   %[[SCALE:.*]] = arith.constant 4.000000e-01 : f64
// CHECK-PASS-NEXT:   %[[OFFSET:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-PASS-NEXT:   %[[SCALED:.*]] = arith.mulf %[[X]], %[[SCALE]] : f64
// CHECK-PASS-NEXT:   %[[T:.*]] = arith.addf %[[SCALED]], %[[OFFSET]] : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 2.000000e+00 : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %[[T]] : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-PASS-COUNT-4: arith.addf
// CHECK-PASS:        %{{.*}} = arith.constant 0.27007526935428111 : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %[[T]], %{{.*}} : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-PASS-NEXT:   %[[RES:.*]] = arith.subf %{{.*}}, %{{.*}} : f64
// CHECK-PASS-NEXT:   func.return %[[RES]] : f64
// CHECK-PASS-NEXT: }

// ----- f64 with op attributes: expanded with op attribute values (degree=4, [-5, 0]) -----

// CHECK-PASS:      func.func @test_exp_f64_attrs(%[[X:.*]]: f64) -> f64 {
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 4.000000e-01 : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 1.000000e+00 : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %[[X]], %{{.*}} : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 2.000000e+00 : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-PASS-COUNT-4: arith.addf
// CHECK-PASS:        %{{.*}} = arith.constant 0.27007526935428111 : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-PASS-NEXT:   %[[RES2:.*]] = arith.subf %{{.*}}, %{{.*}} : f64
// CHECK-PASS-NEXT:   func.return %[[RES2]] : f64
// CHECK-PASS-NEXT: }

// ----- f32 expanded via pass parameter -----

// CHECK-PASS:      func.func @test_exp_f32(%[[X32:.*]]: f32) -> f32 {
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 4.000000e-01 : f32
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 1.000000e+00 : f32
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %[[X32]], %{{.*}} : f32
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 2.000000e+00 : f32
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-PASS-COUNT-4: arith.addf
// CHECK-PASS:        %{{.*}} = arith.constant 0.270075262 : f32
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-PASS-NEXT:   %[[RES32:.*]] = arith.subf %{{.*}}, %{{.*}} : f32
// CHECK-PASS-NEXT:   func.return %[[RES32]] : f32
// CHECK-PASS-NEXT: }

// ----- f16 expanded via pass parameter -----

// CHECK-PASS:      func.func @test_exp_f16(%[[X16:.*]]: f16) -> f16 {
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 3.999020e-01 : f16
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 1.000000e+00 : f16
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %[[X16]], %{{.*}} : f16
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f16
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant 2.000000e+00 : f16
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f16
// CHECK-PASS-COUNT-4: arith.addf
// CHECK-PASS:        %{{.*}} = arith.constant 2.700200e-01 : f16
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f16
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f16
// CHECK-PASS-NEXT:   %[[RES16:.*]] = arith.subf %{{.*}}, %{{.*}} : f16
// CHECK-PASS-NEXT:   func.return %[[RES16]] : f16
// CHECK-PASS-NEXT: }

// ----- vector<2xf32> expanded via pass parameter -----

// CHECK-PASS:      func.func @test_exp_vec(%[[XV:.*]]: vector<2xf32>) -> vector<2xf32> {
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant dense<4.000000e-01> : vector<2xf32>
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %[[XV]], %{{.*}} : vector<2xf32>
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-PASS-NEXT:   %{{.*}} = arith.constant dense<2.000000e+00> : vector<2xf32>
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-PASS-COUNT-4: arith.addf
// CHECK-PASS:        %{{.*}} = arith.constant dense<0.270075262> : vector<2xf32>
// CHECK-PASS-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-PASS-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-PASS-NEXT:   %[[RESV:.*]] = arith.subf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-PASS-NEXT:   func.return %[[RESV]] : vector<2xf32>
// CHECK-PASS-NEXT: }
