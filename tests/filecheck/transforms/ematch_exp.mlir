// RUN: xdsl-opt --split-input-file -p ematch-exp %s | filecheck %s

func.func @test(%x: f64) -> f64 {
    %res = math.exp %x : f64
    func.return %res : f64
}


// CHECK:      func.func @test(%x : f64) -> f64 {
// CHECK-NEXT:   %0 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %1 = arith.divf %x, %0 : f64
// CHECK-NEXT:   %2 = arith.mulf %1, %0 : f64
// CHECK-NEXT:   %3 = arith.addf %0, %2 : f64
// CHECK-NEXT:   %4 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %5 = arith.divf %x, %4 : f64
// CHECK-NEXT:   %6 = arith.mulf %5, %2 : f64
// CHECK-NEXT:   %7 = arith.addf %3, %6 : f64
// CHECK-NEXT:   %res = equivalence.class %res_1, %7 : f64
// CHECK-NEXT:   %res_1 = math.exp %x : f64
// CHECK-NEXT:   func.return %res : f64
// CHECK-NEXT: }

// -----

func.func @test2(%x: f64) -> f64 {
    %c1 = arith.constant 1.000000e+00 : f64
    %res = math.exp %x : f64 
    func.return %res : f64
}
// we want to observe that no new. constant arith.constant 1.000000e+00 : f64 is created
// CHECK:      func.func @test2(%x : f64) -> f64 {
// CHECK-NEXT:   %c1 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.divf %x, %c1 : f64
// CHECK-NEXT:   %1 = arith.mulf %0, %c1 : f64
// CHECK-NEXT:   %2 = arith.addf %c1, %1 : f64
// CHECK-NEXT:   %3 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %4 = arith.divf %x, %3 : f64
// CHECK-NEXT:   %5 = arith.mulf %4, %1 : f64
// CHECK-NEXT:   %6 = arith.addf %2, %5 : f64
// CHECK-NEXT:   %res = equivalence.class %res_1, %6 : f64
// CHECK-NEXT:   %res_1 = math.exp %x : f64
// CHECK-NEXT:   func.return %res : f64
// CHECK-NEXT: }
