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

  // acc_bound is preserved on math.exp by step 1; this pass currently ignores
  // it (the polynomial choice is a fixed dummy) but should still rewrite the op.
  func.func @exp_f32_with_acc_bound(%x: f32) -> f32 {
    %r = math.exp %x {acc_bound = 1.000000e-03 : f32} : f32
    func.return %r : f32
  }
}

// CHECK:      func.func @exp_f32(%[[X32:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[R32:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[X32]] {scheme = "clenshaw", domain_lower = -103.27892990343184 : f64, domain_upper = 0.000000e+00 : f64} : f32
// CHECK-NEXT:   func.return %[[R32]] : f32
// CHECK-NEXT: }

// CHECK:      func.func @exp_f64(%[[X64:.*]]: f64) -> f64 {
// CHECK-NEXT:   %[[R64:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[X64]] {scheme = "clenshaw", domain_lower = -744.44007192138122 : f64, domain_upper = 0.000000e+00 : f64} : f64
// CHECK-NEXT:   func.return %[[R64]] : f64
// CHECK-NEXT: }

// CHECK:      func.func @exp_f32_with_acc_bound(%[[XB:.*]]: f32) -> f32 {
// CHECK-NEXT:   %[[RB:.*]] = polynomial.eval #polynomial.typed_chebyshev_polynomial<[{{.*}}]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[XB]] {scheme = "clenshaw", domain_lower = -103.27892990343184 : f64, domain_upper = 0.000000e+00 : f64} : f32
// CHECK-NEXT:   func.return %[[RB]] : f32
// CHECK-NEXT: }
