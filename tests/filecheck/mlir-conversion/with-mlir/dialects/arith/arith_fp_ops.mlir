// RUN: MLIR_ROUNDTRIP

module {
  %0 = arith.constant 1.0 : f32
  %1 = arith.constant 1.0 : f64
  %2, %3, %4 = "test.op"() : () -> (bf16, f80, f128)
  %5 = arith.negf %0 : f32
  %6 = arith.negf %1 : f64
  %7 = arith.negf %2 : bf16
  %8 = arith.negf %3 : f80
  %9 = arith.negf %4 : f128
}

// CHECK:        %{{.*}} = arith.negf %{{.*}} : f32
// CHECK:        %{{.*}} = arith.negf %{{.*}} : f64
// CHECK:        %{{.*}} = arith.negf %{{.*}} : bf16
// CHECK:        %{{.*}} = arith.negf %{{.*}} : f80
// CHECK:        %{{.*}} = arith.negf %{{.*}} : f128
