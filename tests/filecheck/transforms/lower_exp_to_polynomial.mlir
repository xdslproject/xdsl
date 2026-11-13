// RUN: xdsl-opt -p lower-exp-to-polynomial %s | filecheck %s

builtin.module {
  // No max_bits_lost: pass picks the default (-1 = correctly-rounded target),
  // domain falls back to [underflow, overflow] of f32.
  func.func @exp_f32(%x: f32) -> f32 {
    %r = math.exp %x : f32
    func.return %r : f32
  }

  // No max_bits_lost: same default behavior, but f64 domain.
  func.func @exp_f64(%x: f64) -> f64 {
    %r = math.exp %x : f64
    func.return %r : f64
  }

  // max_bits_lost is preserved on math.exp by step 1 and drives the polynomial
  // degree picked by _choose_polynomial.
  func.func @exp_f32_with_max_bits_lost(%x: f32) -> f32 {
    %r = math.exp %x {max_bits_lost = 2 : i64} : f32
    func.return %r : f32
  }

  // user lower_bound = -2 > underflow (~-103.28) -> kept at -2
  // user upper_bound =  5 < overflow  (~ 88.72)  -> kept at 5
  func.func @exp_f32_clamped_lower(%x: f32) -> f32 {
    %r = math.exp %x {max_bits_lost = 2 : i64, lower_bound = -2.000000e+00 : f32, upper_bound = 5.000000e+00 : f32} : f32
    func.return %r : f32
  }

  // both user bounds inside representable range -> used as-is
  func.func @exp_f32_tight_bounds(%x: f32) -> f32 {
    %r = math.exp %x {max_bits_lost = 2 : i64, lower_bound = -5.000000e-01 : f32, upper_bound = 5.000000e-01 : f32} : f32
    func.return %r : f32
  }
}

// Without max_bits_lost, math.exp is still lowered using the default
// correctly-rounded target and [underflow, overflow] domain.
// CHECK:      func.func @exp_f32(%[[X32:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[R32:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f32>>, %[[X32]] {scheme = "clenshaw", domain_lower = {{.*}} : f32, domain_upper = {{.*}} : f32} : f32
// CHECK-NEXT:   func.return %[[R32]] : f32
// CHECK-NEXT: }

// CHECK:      func.func @exp_f64(%[[X64:.*]]: f64) -> f64 {
// CHECK-NEXT:   %[[R64:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[X64]] {scheme = "clenshaw", domain_lower = {{.*}} : f64, domain_upper = {{.*}} : f64} : f64
// CHECK-NEXT:   func.return %[[R64]] : f64
// CHECK-NEXT: }

// Without explicit lower/upper bounds, the polynomial domain defaults to
// [underflow, overflow] for the precision (f32: ~[-103.28, 88.72]).
// CHECK:      func.func @exp_f32_with_max_bits_lost(%[[XB:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[RB:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f32>>, %[[XB]] {scheme = "clenshaw", domain_lower = {{.*}} : f32, domain_upper = {{.*}} : f32} : f32
// CHECK-NEXT:   func.return %[[RB]] : f32
// CHECK-NEXT: }

// CHECK:      func.func @exp_f32_clamped_lower(%[[XC:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[RC:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f32>>, %[[XC]] {scheme = "clenshaw", domain_lower = -2.000000e+00 : f32, domain_upper = 5.000000e+00 : f32} : f32
// CHECK-NEXT:   func.return %[[RC]] : f32
// CHECK-NEXT: }

// CHECK:      func.func @exp_f32_tight_bounds(%[[XT:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[RT:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f32>>, %[[XT]] {scheme = "clenshaw", domain_lower = -5.000000e-01 : f32, domain_upper = 5.000000e-01 : f32} : f32
// CHECK-NEXT:   func.return %[[RT]] : f32
// CHECK-NEXT: }
