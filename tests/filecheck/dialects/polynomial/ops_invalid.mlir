// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

// domain_lower > domain_upper
%x0 = "test.op"() : () -> f32
%r0 = polynomial.eval_chebyshev %x0 <[1.000000e+00 : f64, 2.000000e+00 : f64], 1.000000e+00 : f64, -1.000000e+00 : f64> : f32

// CHECK: domain_lower (1.0) must be strictly less than domain_upper (-1.0)

// -----

// domain_lower == domain_upper
%x1 = "test.op"() : () -> f32
%r1 = polynomial.eval_chebyshev %x1 <[1.000000e+00 : f64, 2.000000e+00 : f64], 0.000000e+00 : f64, 0.000000e+00 : f64> : f32

// CHECK: domain_lower (0.0) must be strictly less than domain_upper (0.0)

// -----

// degree 0 (single coefficient)
%x2 = "test.op"() : () -> f32
%r2 = polynomial.eval_chebyshev %x2 <[1.000000e+00 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64> : f32

// CHECK: Chebyshev polynomial must have at least degree 1 (got 1 coefficients)
