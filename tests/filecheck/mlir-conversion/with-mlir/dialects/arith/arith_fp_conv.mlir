// RUN: MLIR_ROUNDTRIP

module {
  %0 = arith.constant 1.0 : f16
  %1 = arith.constant 1.0 : f32
  %2 = arith.constant 1.0 : f64
  %bf, %f80v, %f128v = "test.op"() : () -> (bf16, f80, f128)
  %3 = arith.extf %0 : f16 to f32
  %4 = arith.extf %0 : f16 to f64
  %5 = arith.extf %1 : f32 to f64
  %6 = arith.truncf %1 : f32 to f16
  %7 = arith.truncf %2 : f64 to f32
  %8 = arith.truncf %2 : f64 to f16
  %9 = arith.extf %bf : bf16 to f32
  %10 = arith.extf %bf : bf16 to f64
  %11 = arith.extf %1 : f32 to f80
  %12 = arith.extf %2 : f64 to f128
  %13 = arith.truncf %1 : f32 to bf16
  %14 = arith.truncf %2 : f64 to bf16
  %15 = arith.truncf %f80v : f80 to f32
  %16 = arith.truncf %f128v : f128 to f64
}

// CHECK:        %{{.*}} = arith.extf %{{.*}} : f16 to f32
// CHECK:        %{{.*}} = arith.extf %{{.*}} : f16 to f64
// CHECK:        %{{.*}} = arith.extf %{{.*}} : f32 to f64
// CHECK:        %{{.*}} = arith.truncf %{{.*}} : f32 to f16
// CHECK:        %{{.*}} = arith.truncf %{{.*}} : f64 to f32
// CHECK:        %{{.*}} = arith.truncf %{{.*}} : f64 to f16
// CHECK:        %{{.*}} = arith.extf %{{.*}} : bf16 to f32
// CHECK:        %{{.*}} = arith.extf %{{.*}} : bf16 to f64
// CHECK:        %{{.*}} = arith.extf %{{.*}} : f32 to f80
// CHECK:        %{{.*}} = arith.extf %{{.*}} : f64 to f128
// CHECK:        %{{.*}} = arith.truncf %{{.*}} : f32 to bf16
// CHECK:        %{{.*}} = arith.truncf %{{.*}} : f64 to bf16
// CHECK:        %{{.*}} = arith.truncf %{{.*}} : f80 to f32
// CHECK:        %{{.*}} = arith.truncf %{{.*}} : f128 to f64
