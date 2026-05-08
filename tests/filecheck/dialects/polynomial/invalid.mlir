// RUN: xdsl-opt --split-input-file --parsing-diagnostics %s | filecheck %s

"test.op"() {poly_ty = !polynomial.polynomial<ring = 42 : i32>} : () -> ()

// CHECK: expected RingAttr in polynomial type, got 42 : i32

// -----

"test.op"() {x = #polynomial.typed_chebyshev_polynomial #polynomial.chebyshev_polynomial<[1.0 : f64, 2.0 : f64]>} : () -> ()

// CHECK: expected TypedChebyshevPolynomialAttr, got #polynomial.chebyshev_polynomial
