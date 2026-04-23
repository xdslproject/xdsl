// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// CHECK-NEXT:   "test.op"() {poly = #polynomial.chebyshev<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64>} : () -> ()
"test.op"() {poly = #polynomial.chebyshev<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64>} : () -> ()

// CHECK-NEXT:   "test.op"() {poly = #polynomial.chebyshev<[1.000000e+00 : f64, 2.000000e+00 : f64], -1.000000e+01 : f64, 0.000000e+00 : f64>} : () -> ()
"test.op"() {poly = #polynomial.chebyshev<[1.000000e+00 : f64, 2.000000e+00 : f64], -1.000000e+01 : f64, 0.000000e+00 : f64>} : () -> ()

// CHECK-NEXT:   "test.op"() {poly = #polynomial.chebyshev<[4.200000e+01 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64>} : () -> ()
"test.op"() {poly = #polynomial.chebyshev<[4.200000e+01 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64>} : () -> ()

// CHECK-NEXT:   "test.op"() {poly = #polynomial.chebyshev<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64, 4.000000e+00 : f64, 5.000000e+00 : f64], -2.000000e+00 : f64, 2.000000e+00 : f64>} : () -> ()
"test.op"() {poly = #polynomial.chebyshev<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64, 4.000000e+00 : f64, 5.000000e+00 : f64], -2.000000e+00 : f64, 2.000000e+00 : f64>} : () -> ()

// CHECK-NEXT: }
