// RUN: xdsl-opt -p expand-polynomial-eval %s | filecheck %s

builtin.module {
  // f64 with domain bounds [-5, 0], degree 2.
  // Scale = 2/(0 - (-5)) = 0.4, offset = -((-5)+0)/(0-(-5)) = 1.0.
  func.func @clenshaw_f64_with_domain(%x: f64) -> f64 {
    %r = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x {scheme = "clenshaw", domain_lower = -5.000000e+00 : f64, domain_upper = 0.000000e+00 : f64} : f64
    func.return %r : f64
  }

  // f64 without domain bounds: skip the affine remap (t = x).
  func.func @clenshaw_f64_no_domain(%x: f64) -> f64 {
    %r = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x {scheme = "clenshaw"} : f64
    func.return %r : f64
  }

  // f32 with domain [-1, 1]: scale=1.0, offset=0.0.
  func.func @clenshaw_f32(%x: f32) -> f32 {
    %r = polynomial.eval #polynomial.typed_chebyshev_polynomial<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x {scheme = "clenshaw", domain_lower = -1.000000e+00 : f64, domain_upper = 1.000000e+00 : f64} : f32
    func.return %r : f32
  }

  // Vector type, with domain bounds, degree 1.
  func.func @clenshaw_vec(%x: vector<4xf32>) -> vector<4xf32> {
    %r = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x {scheme = "clenshaw", domain_lower = -2.000000e+00 : f64, domain_upper = 2.000000e+00 : f64} : vector<4xf32>
    func.return %r : vector<4xf32>
  }
}

// CHECK: builtin.module {

// ===== f64 with domain [-5, 0], coeffs [1, 2, 3] =====

// CHECK:      func.func @clenshaw_f64_with_domain(%[[X:.*]]: f64) -> f64 {
//             Domain mapping: t = x * 0.4 + 1.0
// CHECK-NEXT:   %[[SCALE:.*]] = arith.constant 4.000000e-01 : f64
// CHECK-NEXT:   %[[OFFSET:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %[[SCALED:.*]] = arith.mulf %[[X]], %[[SCALE]] : f64
// CHECK-NEXT:   %[[T:.*]] = arith.addf %[[SCALED]], %[[OFFSET]] : f64
//             two_t = 2 * t
// CHECK-NEXT:   %[[TWO:.*]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %[[TWO_T:.*]] = arith.mulf %[[TWO]], %[[T]] : f64
//             b_{n+2} = 0, b_{n+1} = 0
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
//             2 Clenshaw iterations
// CHECK-COUNT-2: arith.addf
//             Final: result = c_0/2 + t * b_1 - b_2
// CHECK:        %[[C0_HALF:.*]] = arith.constant 5.000000e-01 : f64
// CHECK-NEXT:   %[[T_B1:.*]] = arith.mulf %[[T]], %{{.*}} : f64
// CHECK-NEXT:   %[[ADD:.*]] = arith.addf %[[C0_HALF]], %[[T_B1]] : f64
// CHECK-NEXT:   %[[RES:.*]] = arith.subf %[[ADD]], %{{.*}} : f64
// CHECK-NEXT:   func.return %[[RES]] : f64
// CHECK-NEXT: }

// ===== f64 without domain bounds: t = x, no scale/offset ops =====

// CHECK:      func.func @clenshaw_f64_no_domain(%[[XN:.*]]: f64) -> f64 {
// CHECK-NEXT:   %[[TWO_N:.*]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %[[TWO_T_N:.*]] = arith.mulf %[[TWO_N]], %[[XN]] : f64
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-COUNT-2: arith.addf
// CHECK:        %[[C0H_N:.*]] = arith.constant 5.000000e-01 : f64
// CHECK-NEXT:   %[[TB1_N:.*]] = arith.mulf %[[XN]], %{{.*}} : f64
// CHECK-NEXT:   %{{.*}} = arith.addf %[[C0H_N]], %[[TB1_N]] : f64
// CHECK-NEXT:   %[[RES_N:.*]] = arith.subf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:   func.return %[[RES_N]] : f64
// CHECK-NEXT: }

// ===== f32 with [-1, 1]: coefficients converted to f32 =====

// CHECK:      func.func @clenshaw_f32(%[[X32:.*]]: f32) -> f32 {
// CHECK-NEXT:   %{{.*}} = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %{{.*}} = arith.constant -0.000000e+00 : f32
// CHECK-NEXT:   %{{.*}} = arith.mulf %[[X32]], %{{.*}} : f32
// CHECK-NEXT:   %[[T32:.*]] = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   %{{.*}} = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %[[T32]] : f32
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK-COUNT-2: arith.addf
// CHECK:        %{{.*}} = arith.constant 2.500000e-01 : f32
// CHECK-NEXT:   %{{.*}} = arith.mulf %[[T32]], %{{.*}} : f32
// CHECK-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   %[[RES32:.*]] = arith.subf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   func.return %[[RES32]] : f32
// CHECK-NEXT: }

// ===== vector<4xf32> with [-2, 2], degree 1 =====
//   scale = 0.5, offset = -0.0 (since (lower+upper) == 0).

// CHECK:      func.func @clenshaw_vec(%[[XV:.*]]: vector<4xf32>) -> vector<4xf32> {
// CHECK-NEXT:   %{{.*}} = arith.constant dense<5.000000e-01> : vector<4xf32>
// CHECK-NEXT:   %{{.*}} = arith.constant dense<-0.000000e+00> : vector<4xf32>
// CHECK-NEXT:   %{{.*}} = arith.mulf %[[XV]], %{{.*}} : vector<4xf32>
// CHECK-NEXT:   %[[TV:.*]] = arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-NEXT:   %{{.*}} = arith.constant dense<2.000000e+00> : vector<4xf32>
// CHECK-NEXT:   %{{.*}} = arith.mulf %{{.*}}, %[[TV]] : vector<4xf32>
// CHECK-NEXT:   %{{.*}} = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-NEXT:   %{{.*}} = arith.constant dense<0.000000e+00> : vector<4xf32>
//             1 Clenshaw iteration (degree=1)
// CHECK-COUNT-1: arith.addf
// CHECK:        %{{.*}} = arith.constant dense<5.000000e-01> : vector<4xf32>
// CHECK-NEXT:   %{{.*}} = arith.mulf %[[TV]], %{{.*}} : vector<4xf32>
// CHECK-NEXT:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-NEXT:   %[[RESV:.*]] = arith.subf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-NEXT:   func.return %[[RESV]] : vector<4xf32>
// CHECK-NEXT: }

// CHECK: }
