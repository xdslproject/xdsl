// RUN: mlir-opt %s | filecheck %s
// RUN: xdsl-opt %s | filecheck %s

// CHECK: module {

// f16 is always representable with scientific format
arith.constant 1.1 : f16
// CHECK-NEXT: {{%.*}} = arith.constant 1.099610e+00 : f16

// f32 is represented by scientific format if it is precise enough
arith.constant 3.1415 : f32
// CHECK-NEXT: {{%.*}} = arith.constant 3.141500e+00 : f32

// else, f32 is printed with 9 significant digits
arith.constant 3.141592 : f32
// CHECK-NEXT: {{%.*}} = arith.constant 3.14159203 : f32

// if the decimal separator is within these 9 siginficant non-zero digits, fine
arith.constant 2.997925e+05 : f32
// CHECK-NEXT: {{%.*}} = arith.constant 299792.5 : f32

// else, print hex format
arith.constant 2.997925e+06 : f32
// CHECK-NEXT: {{%.*}} = arith.constant 0x4A36FA94 : f32

// f64 is represented by scientific format if it is precise enough
arith.constant 3.1415 : f64
// CHECK-NEXT: {{%.*}} = arith.constant 3.141500e+00 : f64

// else, f64 is printed with 17 significant digits
arith.constant 3.141592 : f64
// CHECK-NEXT: {{%.*}} = arith.constant 3.1415920000000002 : f64
