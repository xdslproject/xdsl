// RUN: xdsl-opt -p expand-math-to-chebyshev %s | filecheck %s

builtin.module {
  func.func @test_exp_f64(%x: f64) -> f64 {
    %y = math.exp %x {chebyshev_degree = 12 : i64, lower = -10.0 : f64, upper = 0.0 : f64} : f64
    func.return %y : f64
  }
  func.func @test_exp_f32(%x: f32) -> f32 {
    %y = math.exp %x {chebyshev_degree = 12 : i64, lower = -10.0 : f64, upper = 0.0 : f64} : f32
    func.return %y : f32
  }
  func.func @test_exp_vec(%x: vector<2xf32>) -> vector<2xf32> {
    %y = math.exp %x {chebyshev_degree = 12 : i64, lower = -10.0 : f64, upper = 0.0 : f64} : vector<2xf32>
    func.return %y : vector<2xf32>
  }
  func.func @test_exp_no_attr(%x: f64) -> f64 {
    %y = math.exp %x : f64
    func.return %y : f64
  }
}

// Degree=12 on [-10, 0].
// Structure: domain mapping (scale=0.2, offset=1.0, mulf, addf),
//            2*t precomputation,
//            two zeros for b_{n+2}, b_{n+1},
//            12 Clenshaw iterations (const + mulf + subf + addf each),
//            final combination (c0/2, t*b1, addf, subf).

// CHECK: builtin.module {

// ----- f64 test (degree=12, [-10, 0]) -----
// CHECK:      func.func @test_exp_f64(%[[X:.*]]: f64) -> f64 {
//               Domain mapping: t = x * 0.2 + 1.0
// CHECK-NEXT:   %[[SCALE:.*]] = arith.constant 2.000000e-01 : f64
// CHECK-NEXT:   %[[OFFSET:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %[[SCALED:.*]] = arith.mulf %[[X]], %[[SCALE]] : f64
// CHECK-NEXT:   %[[T:.*]] = arith.addf %[[SCALED]], %[[OFFSET]] : f64
//               2*t
// CHECK-NEXT:   %[[TWO:.*]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %[[TWO_T:.*]] = arith.mulf %[[TWO]], %[[T]] : f64
//               b_{n+2} = 0, b_{n+1} = 0
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
//               12 Clenshaw iterations
// CHECK-COUNT-12: arith.addf
//               Final: result = c_0/2 + t * b_1 - b_2
// CHECK:        %[[C0_HALF:.*]] = arith.constant 0.1835408126093285 : f64
// CHECK-NEXT:   %[[T_B1:.*]] = arith.mulf %[[T]], %{{.*}} : f64
// CHECK-NEXT:   %[[ADD:.*]] = arith.addf %[[C0_HALF]], %[[T_B1]] : f64
// CHECK-NEXT:   %[[RES:.*]] = arith.subf %[[ADD]], %{{.*}} : f64
// CHECK-NEXT:   func.return %[[RES]] : f64
// CHECK-NEXT: }

// ----- f32 test -----
// CHECK:      func.func @test_exp_f32(%[[X32:.*]]: f32) -> f32 {
// CHECK-NEXT:   %{{.*}} = arith.constant 2.000000e-01 : f32
// CHECK-NEXT:   %{{.*}} = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %{{.*}} = arith.mulf %[[X32]], %{{.*}} : f32
// CHECK-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   %{{.*}} = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK:        %{{.*}} = arith.constant 0.183540806 : f32
// CHECK-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   %[[RES32:.*]] = arith.subf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   func.return %[[RES32]] : f32
// CHECK-NEXT: }

// ----- vector<2xf32> test -----
// CHECK:      func.func @test_exp_vec(%[[XV:.*]]: vector<2xf32>) -> vector<2xf32> {
// CHECK-NEXT:   %{{.*}} = arith.constant dense<2.000000e-01> : vector<2xf32>
// CHECK-NEXT:   %{{.*}} = arith.constant dense<1.000000e+00> : vector<2xf32>
// CHECK-NEXT:   %{{.*}} = arith.mulf %[[XV]], %{{.*}} : vector<2xf32>
// CHECK-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-NEXT:   %{{.*}} = arith.constant dense<2.000000e+00> : vector<2xf32>
// CHECK-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK:        %{{.*}} = arith.constant dense<0.183540806> : vector<2xf32>
// CHECK-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-NEXT:   %[[RESV:.*]] = arith.subf %{{.*}}, %{{.*}} : vector<2xf32>
// CHECK-NEXT:   func.return %[[RESV]] : vector<2xf32>
// CHECK-NEXT: }

// ----- no chebyshev_degree attribute: pass should not fire -----
// CHECK:      func.func @test_exp_no_attr(%[[XNA:.*]]: f64) -> f64 {
// CHECK-NEXT:   %[[YNA:.*]] = math.exp %[[XNA]] : f64
// CHECK-NEXT:   func.return %[[YNA]] : f64
// CHECK-NEXT: }
