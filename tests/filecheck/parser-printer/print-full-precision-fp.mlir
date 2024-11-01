// RUN: xdsl-opt %s --split-input-file | filecheck %s
// RUN: xdsl-opt %s --print-full-precision-fp --split-input-file | filecheck %s --check-prefix FULL-PRECISION


// CHECK:            %0 = arith.constant 3.141593e+00 : f32
// FULL-PRECISION:   %0 = arith.constant 3.141592653589793 : f32

%0 = arith.constant 3.141592653589793 : f32
