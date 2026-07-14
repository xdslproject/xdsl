// RUN: $XDSL_MLIR_OPT %s | filecheck %s
// RUN: xdsl-opt %s | filecheck %s

// The reduced-precision float types pack/unpack bit-exactly through the
// reduced-float codec, so xdsl-opt and mlir-opt narrow every literal to the
// same constant. Values are either exactly representable or narrow to a value
// whose decimal printing agrees between both tools.

// CHECK: module {

arith.constant 1.5 : tf32
// CHECK-NEXT: {{%.*}} = arith.constant 1.500000e+00 : tf32

arith.constant 1.5 : f8E5M2
// CHECK-NEXT: {{%.*}} = arith.constant 1.500000e+00 : f8E5M2

// 0.1 is not representable in f8E5M2; it narrows to 0.09375.
arith.constant 0.1 : f8E5M2
// CHECK-NEXT: {{%.*}} = arith.constant 9.375000e-02 : f8E5M2

arith.constant 1.25 : f8E4M3
// CHECK-NEXT: {{%.*}} = arith.constant 1.250000e+00 : f8E4M3

arith.constant 1.0625 : f8E3M4
// CHECK-NEXT: {{%.*}} = arith.constant 1.062500e+00 : f8E3M4

arith.constant 1.25 : f8E4M3FN
// CHECK-NEXT: {{%.*}} = arith.constant 1.250000e+00 : f8E4M3FN

arith.constant 0.5 : f8E5M2FNUZ
// CHECK-NEXT: {{%.*}} = arith.constant 5.000000e-01 : f8E5M2FNUZ

arith.constant 1.25 : f8E4M3FNUZ
// CHECK-NEXT: {{%.*}} = arith.constant 1.250000e+00 : f8E4M3FNUZ

arith.constant 0.25 : f8E4M3B11FNUZ
// CHECK-NEXT: {{%.*}} = arith.constant 2.500000e-01 : f8E4M3B11FNUZ

arith.constant 4.0 : f8E8M0FNU
// CHECK-NEXT: {{%.*}} = arith.constant 4.000000e+00 : f8E8M0FNU

// f8E8M0FNU has no mantissa; 3.0 snaps to the nearest power of two, 4.0.
arith.constant 3.0 : f8E8M0FNU
// CHECK-NEXT: {{%.*}} = arith.constant 4.000000e+00 : f8E8M0FNU

arith.constant 1.375 : f6E2M3FN
// CHECK-NEXT: {{%.*}} = arith.constant 1.375000e+00 : f6E2M3FN

// 0.1 is not representable in f6E2M3FN; it narrows to 0.125.
arith.constant 0.1 : f6E2M3FN
// CHECK-NEXT: {{%.*}} = arith.constant 1.250000e-01 : f6E2M3FN

arith.constant 1.5 : f6E3M2FN
// CHECK-NEXT: {{%.*}} = arith.constant 1.500000e+00 : f6E3M2FN

arith.constant 1.5 : f4E2M1FN
// CHECK-NEXT: {{%.*}} = arith.constant 1.500000e+00 : f4E2M1FN

// 0.1 underflows f4E2M1FN's subnormals and narrows to 0.0.
arith.constant 0.1 : f4E2M1FN
// CHECK-NEXT: {{%.*}} = arith.constant 0.000000e+00 : f4E2M1FN
