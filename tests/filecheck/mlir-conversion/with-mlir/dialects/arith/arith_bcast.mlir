// RUN: MLIR_ROUNDTRIP

module {
  %0, %4, %7, %12, %15 = "test.op"() : () -> (i32, i64, i16, i80, i128)
  %1 = arith.bitcast %0 : i32 to i32
  %2 = arith.bitcast %0 : i32 to f32
  %3 = arith.bitcast %2 : f32 to i32

  %5 = arith.bitcast %4 : i64 to f64
  %6 = arith.bitcast %5 : f64 to i64

  %8 = arith.bitcast %7 : i16 to bf16
  %9 = arith.bitcast %8 : bf16 to i16
  %10 = arith.bitcast %8 : bf16 to f16
  %11 = arith.bitcast %10 : f16 to bf16

  %13 = arith.bitcast %12 : i80 to f80
  %14 = arith.bitcast %13 : f80 to i80

  %16 = arith.bitcast %15 : i128 to f128
  %17 = arith.bitcast %16 : f128 to i128
}

// CHECK:        %{{.*}} = arith.bitcast %{{.*}} : i32 to i32
// CHECK-NEXT:   %{{.*}} = arith.bitcast %{{.*}} : i32 to f32
// CHECK-NEXT:   %{{.*}} = arith.bitcast %{{.*}} : f32 to i32

// CHECK:        %{{.*}} = arith.bitcast %{{.*}} : i64 to f64
// CHECK-NEXT:   %{{.*}} = arith.bitcast %{{.*}} : f64 to i64

// CHECK:        %{{.*}} = arith.bitcast %{{.*}} : i16 to bf16
// CHECK-NEXT:   %{{.*}} = arith.bitcast %{{.*}} : bf16 to i16
// CHECK-NEXT:   %{{.*}} = arith.bitcast %{{.*}} : bf16 to f16
// CHECK-NEXT:   %{{.*}} = arith.bitcast %{{.*}} : f16 to bf16

// CHECK:        %{{.*}} = arith.bitcast %{{.*}} : i80 to f80
// CHECK-NEXT:   %{{.*}} = arith.bitcast %{{.*}} : f80 to i80

// CHECK:        %{{.*}} = arith.bitcast %{{.*}} : i128 to f128
// CHECK-NEXT:   %{{.*}} = arith.bitcast %{{.*}} : f128 to i128
