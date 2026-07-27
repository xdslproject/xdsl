// RUN: MLIR_ROUNDTRIP

module {
  %0 = arith.constant 1.0 : f64
  %1 = arith.constant 1.0 : f64
  %2 = arith.constant 1 : i32
  %3 = arith.constant 2 : i32
  %4 = arith.constant 1 : index
  %5 = arith.constant 2 : index
  %bf_lhs, %bf_rhs, %f80_lhs, %f80_rhs, %f128_lhs, %f128_rhs = "test.op"() : () -> (bf16, bf16, f80, f80, f128, f128)
  %6 = arith.cmpf ogt, %0, %1 : f64
  %7 = arith.select %6, %0, %1 : f64
  %8 = arith.cmpi ne, %2, %3 : i32
  %9 = arith.select %8, %2, %3 : i32
  %10 = arith.cmpi ne, %4, %5 : index
  %11 = arith.select %10, %4, %5 : index
  %12 = arith.cmpf ogt, %bf_lhs, %bf_rhs : bf16
  %13 = arith.select %12, %bf_lhs, %bf_rhs : bf16
  %14 = arith.cmpf ogt, %f80_lhs, %f80_rhs : f80
  %15 = arith.select %14, %f80_lhs, %f80_rhs : f80
  %16 = arith.cmpf ogt, %f128_lhs, %f128_rhs : f128
  %17 = arith.select %16, %f128_lhs, %f128_rhs : f128
}

// CHECK:        %{{.*}} = arith.cmpf ogt, %{{.*}}, %{{.*}} : f64
// CHECK:        %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : f64
// CHECK:        %{{.*}} = arith.cmpi ne, %{{.*}}, %{{.*}} : i32
// CHECK:        %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// CHECK:        %{{.*}} = arith.cmpi ne, %{{.*}}, %{{.*}} : index
// CHECK:        %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK:        %{{.*}} = arith.cmpf ogt, %{{.*}}, %{{.*}} : bf16
// CHECK:        %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : bf16
// CHECK:        %{{.*}} = arith.cmpf ogt, %{{.*}}, %{{.*}} : f80
// CHECK:        %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : f80
// CHECK:        %{{.*}} = arith.cmpf ogt, %{{.*}}, %{{.*}} : f128
// CHECK:        %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : f128
