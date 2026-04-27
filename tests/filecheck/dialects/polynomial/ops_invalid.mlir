// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

// domain_lower > domain_upper
%x0 = "test.op"() : () -> f32
%r0 = polynomial.eval <[1.000000e+00 : f64, 2.000000e+00 : f64]>, %x0 {scheme = "clenshaw", domain_lower = 1.000000e+00 : f64, domain_upper = -1.000000e+00 : f64} : f32

// CHECK: domain_lower (1.0) must be strictly less than domain_upper (-1.0)

// -----

// domain_lower == domain_upper
%x1 = "test.op"() : () -> f32
%r1 = polynomial.eval <[1.000000e+00 : f64, 2.000000e+00 : f64]>, %x1 {scheme = "clenshaw", domain_lower = 0.000000e+00 : f64, domain_upper = 0.000000e+00 : f64} : f32

// CHECK: domain_lower (0.0) must be strictly less than domain_upper (0.0)

// -----

// degree 0 (single coefficient)
%x2 = "test.op"() : () -> f32
%r2 = polynomial.eval <[1.000000e+00 : f64]>, %x2 {scheme = "clenshaw"} : f32

// CHECK: Chebyshev polynomial must have at least degree 1 (got 1 coefficients)

// -----

// unknown evaluation scheme
%x3 = "test.op"() : () -> f32
%r3 = polynomial.eval <[1.000000e+00 : f64, 2.000000e+00 : f64]>, %x3 {scheme = "horner"} : f32

// CHECK: unknown evaluation scheme 'horner'
