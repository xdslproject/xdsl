// RUN: xdsl-opt %s --print-reduced-precision --split-input-file | filecheck %s --check-prefix REDUCED-PRECISION
// RUN: xdsl-opt %s --split-input-file | filecheck %s --check-prefix FULL-PRECISION
// RUN: xdsl-opt %s --split-input-file | xdsl-opt %s | filecheck %s --check-prefix ROUNDTRIP

// REDUCED-PRECISION:  builtin.module {
// FULL-PRECISION:     builtin.module {
// ROUNDTRIP:          builtin.module {
builtin.module {

// REDUCED-PRECISION:  %0 = arith.constant 3.141593e+00 : f64
// FULL-PRECISION:     %0 = arith.constant 3.141592653589793 : f64
// ROUNDTRIP:          %0 = arith.constant 3.141592653589793 : f64

%0 = arith.constant 3.141592653589793 : f64

"test.op"(%0) : (f64) -> ()
}
