// RUN: xdsl-opt %s --print-reduced-precision-fp --split-input-file | filecheck %s
// RUN: xdsl-opt %s --split-input-file | filecheck %s --check-prefix FULL-PRECISION
// RUN: xdsl-opt %s --split-input-file -p "canonicalize,mlir-opt[canonicalize],canonicalize" | filecheck %s --check-prefix MLIR-AND-BACK

// CHECK:            builtin.module {
// FULL-PRECISION:   builtin.module {
// MLIR-AND-BACK:    builtin.module {
builtin.module {

// CHECK:            %0 = arith.constant 3.141593e+00 : f64
// FULL-PRECISION:   %0 = arith.constant 3.141592653589793 : f64
// MLIR-AND-BACK:    %0 = arith.constant 3.141592653589793 : f64

%0 = arith.constant 3.141592653589793 : f64

"test.op"(%0) : (f64) -> ()
}
