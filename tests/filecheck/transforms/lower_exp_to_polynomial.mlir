// RUN: xdsl-opt -p lower-exp-to-polynomial %s | filecheck %s

builtin.module {
  func.func @exp_f32(%x: f32) -> f32 {
    %r = math.exp %x : f32
    func.return %r : f32
  }

  func.func @exp_f64(%x: f64) -> f64 {
    %r = math.exp %x : f64
    func.return %r : f64
  }

  // acc_bound is preserved on math.exp by step 1 and drives the polynomial
  // degree picked by _choose_polynomial.
  func.func @exp_f32_with_acc_bound(%x: f32) -> f32 {
    %r = math.exp %x {acc_bound = 1.000000e-03 : f64} : f32
    func.return %r : f32
  }

  // user lower_bound = -2 < underflow (-1) -> clamped to -1
  // user upper_bound =  5 < overflow (~88.72) -> kept at 5
  func.func @exp_f32_clamped_lower(%x: f32) -> f32 {
    %r = math.exp %x {acc_bound = 1.000000e-03 : f64, lower_bound = -2.000000e+00 : f32, upper_bound = 5.000000e+00 : f32} : f32
    func.return %r : f32
  }

  // both user bounds inside representable range -> used as-is
  func.func @exp_f32_tight_bounds(%x: f32) -> f32 {
    %r = math.exp %x {acc_bound = 1.000000e-03 : f64, lower_bound = -5.000000e-01 : f32, upper_bound = 5.000000e-01 : f32} : f32
    func.return %r : f32
  }
}

// Without acc_bound, math.exp is left untouched.
// CHECK:      func.func @exp_f32(%[[X32:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[R32:.*]] = math.exp %[[X32]] : f32
// CHECK-NEXT:   func.return %[[R32]] : f32
// CHECK-NEXT: }

// CHECK:      func.func @exp_f64(%[[X64:.*]]: f64) -> f64 {
// CHECK-NEXT:   %[[R64:.*]] = math.exp %[[X64]] : f64
// CHECK-NEXT:   func.return %[[R64]] : f64
// CHECK-NEXT: }

// Without explicit lower/upper bounds, the polynomial domain defaults to
// [underflow, overflow] for the precision (f32 overflow ~= 88.72).
// CHECK:      func.func @exp_f32_with_acc_bound(%[[XB:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[RB:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[XB]] {scheme = "clenshaw", domain_lower = -1.000000e+00 : f64, domain_upper = {{.*}} : f64} : f32
// CHECK-NEXT:   func.return %[[RB]] : f32
// CHECK-NEXT: }

// CHECK:      func.func @exp_f32_clamped_lower(%[[XC:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[RC:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[XC]] {scheme = "clenshaw", domain_lower = -1.000000e+00 : f64, domain_upper = 5.000000e+00 : f64} : f32
// CHECK-NEXT:   func.return %[[RC]] : f32
// CHECK-NEXT: }

// CHECK:      func.func @exp_f32_tight_bounds(%[[XT:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[RT:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[XT]] {scheme = "clenshaw", domain_lower = -5.000000e-01 : f64, domain_upper = 5.000000e-01 : f64} : f32
// CHECK-NEXT:   func.return %[[RT]] : f32
// CHECK-NEXT: }
