// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

"builtin.module"() ({
  %lhsi1 = "test.op"() : () -> (i1)
  %lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
  %lhsi64 = "test.op"() : () -> (i64)
  %lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
  %lhsf64 = "test.op"() : () -> (f64)
  %lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)


  %divsi = arith.divsi %lhsi32, %rhsi32 : i32
  %divui = arith.divui %lhsi32, %rhsi32 : i32

  // CHECK:      %6 = arith.divsi %1#0, %1#1 : i32
  // CHECK-NEXT: %7 = arith.divui %1#0, %1#1 : i32

  %ceildivsi = arith.ceildivsi %lhsi32, %rhsi32 : i32
  %floordivsi = arith.floordivsi %lhsi32, %rhsi32 : i32
  %ceildivui = arith.ceildivui %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: %8 = arith.ceildivsi %1#0, %1#1 : i32
  // CHECK-NEXT: %9 = arith.floordivsi %1#0, %1#1 : i32
  // CHECK-NEXT: %10 = arith.ceildivui %1#0, %1#1 : i32

  %remsi = arith.remsi %lhsi32, %rhsi32 : i32
  %remui = arith.remui %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: %11 = arith.remsi %1#0, %1#1 : i32
  // CHECK-NEXT: %12 = arith.remui %1#0, %1#1 : i32

  %maxsi = arith.maxsi %lhsi32, %rhsi32 : i32
  %minsi = arith.minsi %lhsi32, %rhsi32 : i32
  %maxui = arith.maxui %lhsi32, %rhsi32 : i32
  %minui = arith.minui %lhsi32, %rhsi32 : i32
  
  // CHECK-NEXT: %13 = arith.maxsi %1#0, %1#1 : i32
  // CHECK-NEXT: %14 = arith.minsi %1#0, %1#1 : i32
  // CHECK-NEXT: %15 = arith.maxui %1#0, %1#1 : i32
  // CHECK-NEXT: %16 = arith.minui %1#0, %1#1 : i32

  %shli = arith.shli %lhsi32, %rhsi32 : i32
  %shrui = arith.shrui %lhsi32, %rhsi32 : i32
  %shrsi = arith.shrsi %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: %17 = arith.shli %1#0, %1#1 : i32
  // CHECK-NEXT: %18 = arith.shrui %1#0, %1#1 : i32
  // CHECK-NEXT: %19 = arith.shrsi %1#0, %1#1 : i32

  %cmpi = arith.cmpi slt, %lhsi32, %rhsi32 : i32
  %cmpf = arith.cmpf ogt, %lhsf32, %rhsf32 : f32

  // CHECK-NEXT: %20 = arith.cmpi slt, %1#0, %1#1 : i32
  // CHECK-NEXT: %21 = arith.cmpf ogt, %3#0, %3#1 : f32

  %maxf = arith.maxf %lhsf32, %rhsf32 : f32
  %maxf_vector = arith.maxf %lhsvec, %rhsvec : vector<4xf32>
  %minf = arith.minf %lhsf32, %rhsf32 : f32
  %minf_vector = arith.minf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %22 = arith.maxf %3#0, %3#1 : f32
  // CHECK-NEXT: %23 = arith.maxf %5#0, %5#1 : vector<4xf32>
  // CHECK-NEXT: %24 = arith.minf %3#0, %3#1 : f32
  // CHECK-NEXT: %25 = arith.minf %5#0, %5#1 : vector<4xf32>

  %addf = arith.addf %lhsf32, %rhsf32 : f32
  %addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %26 = arith.addf %3#0, %3#1 : f32
  // CHECK-NEXT: %27 = arith.addf %5#0, %5#1 : vector<4xf32>

  %subf = arith.subf %lhsf32, %rhsf32 : f32
  %subf_vector = arith.subf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %28 = arith.subf %3#0, %3#1 : f32
  // CHECK-NEXT: %29 = arith.subf %5#0, %5#1 : vector<4xf32>

  %mulf = arith.mulf %lhsf32, %rhsf32 : f32
  %mulf_vector = arith.mulf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %30 = arith.mulf %3#0, %3#1 : f32
  // CHECK-NEXT: %31 = arith.mulf %5#0, %5#1 : vector<4xf32>

  %divf = arith.divf %lhsf32, %rhsf32 : f32
  %divf_vector = arith.divf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %32 = arith.divf %3#0, %3#1 : f32
  // CHECK-NEXT: %33 = arith.divf %5#0, %5#1 : vector<4xf32>

  %negf = arith.negf %lhsf32 : f32

  // CHECK-NEXT: %34 = arith.negf %3#0 : f32

  %extf = arith.extf %lhsf32 : f32 to f64
  %extui = arith.extui %lhsi32 : i32 to i64
  %truncf = arith.truncf %lhsf64 : f64 to f32
  %trunci = arith.trunci %lhsi64 : i64 to i32

  // CHECK-NEXT: %35 = arith.extf %3#0 : f32 to f64
  // CHECK-NEXT: %36 = arith.extui %1#0 : i32 to i64
  // CHECK-NEXT: %37 = arith.truncf %4 : f64 to f32
  // CHECK-NEXT: %38 = arith.trunci %2 : i64 to i32

  %selecti = arith.select %lhsi1, %lhsi32, %rhsi32 : i32
  %selectf = arith.select %lhsi1, %lhsf32, %rhsf32 : f32

  // CHECK-NEXT: %39 = arith.select %0, %1#0, %1#1 : i32
  // CHECK-NEXT: %40 = arith.select %0, %3#0, %3#1 : f32
}) : () -> ()
