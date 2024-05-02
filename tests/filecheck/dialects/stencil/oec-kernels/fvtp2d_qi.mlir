// RUN: XDSL_ROUNDTRIP
// RUN: xdsl-opt %s -p stencil-storage-materialization,stencil-shape-inference | filecheck %s --check-prefix SHAPE

func.func @fvtp2d_qi(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<?x?x?xf64>, %arg5: !stencil.field<?x?x?xf64>, %arg6: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %2 = stencil.cast %arg2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %3 = stencil.cast %arg3 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %4 = stencil.cast %arg4 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %5 = stencil.cast %arg5 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %6 = stencil.cast %arg6 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  %7 = stencil.load %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
  %8 = stencil.load %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
  %9 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
  %10 = stencil.load %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
  %11 = stencil.load %4 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
  %12 = stencil.apply (%arg7 = %7 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 7.000000e+00 : f64
    %cst_1 = arith.constant 1.200000e+01 : f64
    %17 = arith.divf %cst_0, %cst_1 : f64
    %18 = arith.divf %cst, %cst_1 : f64
    %19 = stencil.access %arg7 [0, -1, 0] : !stencil.temp<?x?x?xf64>
    %20 = stencil.access %arg7 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %21 = arith.addf %19, %20 : f64
    %22 = stencil.access %arg7 [0, -2, 0] : !stencil.temp<?x?x?xf64>
    %23 = stencil.access %arg7 [0, 1, 0] : !stencil.temp<?x?x?xf64>
    %24 = arith.addf %22, %23 : f64
    %25 = arith.mulf %17, %21 : f64
    %26 = arith.mulf %18, %24 : f64
    %27 = arith.addf %25, %26 : f64
    %28 = stencil.store_result %27 : !stencil.result<f64>
    stencil.return %28 : !stencil.result<f64>
  }
  %13:4 = stencil.apply (%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %12 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %17 = stencil.access %arg8 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %18 = stencil.access %arg7 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %19 = arith.subf %17, %18 : f64
    %20 = stencil.access %arg8 [0, 1, 0] : !stencil.temp<?x?x?xf64>
    %21 = arith.subf %20, %18 : f64
    %22 = arith.addf %19, %21 : f64
    %23 = arith.mulf %19, %21 : f64
    %24 = arith.cmpf olt, %23, %cst : f64
    %25 = arith.select %24, %cst_0, %cst : f64
    %26 = stencil.store_result %19 : !stencil.result<f64>
    %27 = stencil.store_result %21 : !stencil.result<f64>
    %28 = stencil.store_result %22 : !stencil.result<f64>
    %29 = stencil.store_result %25 : !stencil.result<f64>
    stencil.return %26, %27, %28, %29 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  }
  %14 = stencil.apply (%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %13#0 : !stencil.temp<?x?x?xf64>, %arg10 = %13#1 : !stencil.temp<?x?x?xf64>, %arg11 = %13#2 : !stencil.temp<?x?x?xf64>, %arg12 = %13#3 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %17 = stencil.access %arg12 [0, -1, 0] : !stencil.temp<?x?x?xf64>
    %18 = arith.cmpf oeq, %17, %cst : f64
    %19 = arith.select %18, %cst_0, %cst : f64
    %20 = stencil.access %arg12 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %21 = arith.mulf %20, %19 : f64
    %22 = arith.addf %17, %21 : f64
    %23 = stencil.access %arg8 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %24 = arith.cmpf ogt, %23, %cst : f64
    %25 = "scf.if"(%24) ({
      %29 = stencil.access %arg10 [0, -1, 0] : !stencil.temp<?x?x?xf64>
      %30 = stencil.access %arg11 [0, -1, 0] : !stencil.temp<?x?x?xf64>
      %31 = arith.mulf %23, %30 : f64
      %32 = arith.subf %29, %31 : f64
      %33 = arith.subf %cst_0, %23 : f64
      %34 = arith.mulf %33, %32 : f64
      scf.yield %34 : f64
    }, {
      %29 = stencil.access %arg9 [0, 0, 0] : !stencil.temp<?x?x?xf64>
      %30 = stencil.access %arg11 [0, 0, 0] : !stencil.temp<?x?x?xf64>
      %31 = arith.mulf %23, %30 : f64
      %32 = arith.addf %29, %31 : f64
      %33 = arith.addf %cst_0, %23 : f64
      %34 = arith.mulf %33, %32 : f64
      scf.yield %34 : f64
    }) : (i1) -> (f64)
    %26 = arith.mulf %25, %22 : f64
    %27 = "scf.if"(%24) ({
      %29 = stencil.access %arg7 [0, -1, 0] : !stencil.temp<?x?x?xf64>
      %30 = arith.addf %29, %26 : f64
      scf.yield %30 : f64
    }, {
      %29 = stencil.access %arg7 [0, 0, 0] : !stencil.temp<?x?x?xf64>
      %30 = arith.addf %29, %26 : f64
      scf.yield %30 : f64
    }) : (i1) -> (f64)
    %28 = stencil.store_result %27 : !stencil.result<f64>
    stencil.return %28 : !stencil.result<f64>
  }
  %15 = stencil.apply (%arg7 = %10 : !stencil.temp<?x?x?xf64>, %arg8 = %14 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
    %17 = stencil.access %arg7 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %18 = stencil.access %arg8 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %19 = arith.mulf %17, %18 : f64
    %20 = stencil.store_result %19 : !stencil.result<f64>
    stencil.return %20 : !stencil.result<f64>
  }
  %16 = stencil.apply (%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %11 : !stencil.temp<?x?x?xf64>, %arg9 = %15 : !stencil.temp<?x?x?xf64>, %arg10 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
    %17 = stencil.access %arg7 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %18 = stencil.access %arg8 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %19 = arith.mulf %17, %18 : f64
    %20 = stencil.access %arg9 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %21 = stencil.access %arg9 [0, 1, 0] : !stencil.temp<?x?x?xf64>
    %22 = arith.subf %20, %21 : f64
    %23 = arith.addf %19, %22 : f64
    %24 = stencil.access %arg10 [0, 0, 0] : !stencil.temp<?x?x?xf64>
    %25 = arith.divf %23, %24 : f64
    %26 = stencil.store_result %25 : !stencil.result<f64>
    stencil.return %26 : !stencil.result<f64>
  }
  stencil.store %14 to %6([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  stencil.store %16 to %5([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  return
}

// CHECK:         func.func @fvtp2d_qi(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>, %arg3 : !stencil.field<?x?x?xf64>, %arg4 : !stencil.field<?x?x?xf64>, %arg5 : !stencil.field<?x?x?xf64>, %arg6 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:      %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %2 = stencil.cast %arg2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %3 = stencil.cast %arg3 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %4 = stencil.cast %arg4 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %5 = stencil.cast %arg5 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %6 = stencil.cast %arg6 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %7 = stencil.load %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:      %8 = stencil.load %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:      %9 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:      %10 = stencil.load %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:      %11 = stencil.load %4 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:      %12 = stencil.apply(%arg7 = %7 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %cst_0 = arith.constant 7.000000e+00 : f64
// CHECK-NEXT:        %cst_1 = arith.constant 1.200000e+01 : f64
// CHECK-NEXT:        %13 = arith.divf %cst_0, %cst_1 : f64
// CHECK-NEXT:        %14 = arith.divf %cst, %cst_1 : f64
// CHECK-NEXT:        %15 = stencil.access %arg7[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %16 = stencil.access %arg7[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %17 = arith.addf %15, %16 : f64
// CHECK-NEXT:        %18 = stencil.access %arg7[0, -2, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %19 = stencil.access %arg7[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %20 = arith.addf %18, %19 : f64
// CHECK-NEXT:        %21 = arith.mulf %13, %17 : f64
// CHECK-NEXT:        %22 = arith.mulf %14, %20 : f64
// CHECK-NEXT:        %23 = arith.addf %21, %22 : f64
// CHECK-NEXT:        %24 = stencil.store_result %23 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %24 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %25, %26, %27, %28 = stencil.apply(%arg7_1 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %12 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %cst_1_1 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:        %cst_0_1 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %29 = stencil.access %arg8[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %30 = stencil.access %arg7_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %31 = arith.subf %29, %30 : f64
// CHECK-NEXT:        %32 = stencil.access %arg8[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %33 = arith.subf %32, %30 : f64
// CHECK-NEXT:        %34 = arith.addf %31, %33 : f64
// CHECK-NEXT:        %35 = arith.mulf %31, %33 : f64
// CHECK-NEXT:        %36 = arith.cmpf olt, %35, %cst_1_1 : f64
// CHECK-NEXT:        %37 = arith.select %36, %cst_0_1, %cst_1_1 : f64
// CHECK-NEXT:        %38 = stencil.store_result %31 : !stencil.result<f64>
// CHECK-NEXT:        %39 = stencil.store_result %33 : !stencil.result<f64>
// CHECK-NEXT:        %40 = stencil.store_result %34 : !stencil.result<f64>
// CHECK-NEXT:        %41 = stencil.store_result %37 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %38, %39, %40, %41 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %42 = stencil.apply(%arg7_2 = %7 : !stencil.temp<?x?x?xf64>, %arg8_1 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %25 : !stencil.temp<?x?x?xf64>, %arg10 = %26 : !stencil.temp<?x?x?xf64>, %arg11 = %27 : !stencil.temp<?x?x?xf64>, %arg12 = %28 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %cst_2 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:        %cst_0_2 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %43 = stencil.access %arg12[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %44 = arith.cmpf oeq, %43, %cst_2 : f64
// CHECK-NEXT:        %45 = arith.select %44, %cst_0_2, %cst_2 : f64
// CHECK-NEXT:        %46 = stencil.access %arg12[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %47 = arith.mulf %46, %45 : f64
// CHECK-NEXT:        %48 = arith.addf %43, %47 : f64
// CHECK-NEXT:        %49 = stencil.access %arg8_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %50 = arith.cmpf ogt, %49, %cst_2 : f64
// CHECK-NEXT:        %51 = "scf.if"(%50) ({
// CHECK-NEXT:          %52 = stencil.access %arg10[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %53 = stencil.access %arg11[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %54 = arith.mulf %49, %53 : f64
// CHECK-NEXT:          %55 = arith.subf %52, %54 : f64
// CHECK-NEXT:          %56 = arith.subf %cst_0_2, %49 : f64
// CHECK-NEXT:          %57 = arith.mulf %56, %55 : f64
// CHECK-NEXT:          scf.yield %57 : f64
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %58 = stencil.access %arg9[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %59 = stencil.access %arg11[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %60 = arith.mulf %49, %59 : f64
// CHECK-NEXT:          %61 = arith.addf %58, %60 : f64
// CHECK-NEXT:          %62 = arith.addf %cst_0_2, %49 : f64
// CHECK-NEXT:          %63 = arith.mulf %62, %61 : f64
// CHECK-NEXT:          scf.yield %63 : f64
// CHECK-NEXT:        }) : (i1) -> f64
// CHECK-NEXT:        %64 = arith.mulf %51, %48 : f64
// CHECK-NEXT:        %65 = "scf.if"(%50) ({
// CHECK-NEXT:          %66 = stencil.access %arg7_2[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %67 = arith.addf %66, %64 : f64
// CHECK-NEXT:          scf.yield %67 : f64
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %68 = stencil.access %arg7_2[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %69 = arith.addf %68, %64 : f64
// CHECK-NEXT:          scf.yield %69 : f64
// CHECK-NEXT:        }) : (i1) -> f64
// CHECK-NEXT:        %70 = stencil.store_result %65 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %70 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %71 = stencil.apply(%arg7_3 = %10 : !stencil.temp<?x?x?xf64>, %arg8_2 = %42 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %72 = stencil.access %arg7_3[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %73 = stencil.access %arg8_2[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %74 = arith.mulf %72, %73 : f64
// CHECK-NEXT:        %75 = stencil.store_result %74 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %75 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %76 = stencil.apply(%arg7_4 = %7 : !stencil.temp<?x?x?xf64>, %arg8_3 = %11 : !stencil.temp<?x?x?xf64>, %arg9_1 = %71 : !stencil.temp<?x?x?xf64>, %arg10_1 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %77 = stencil.access %arg7_4[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %78 = stencil.access %arg8_3[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %79 = arith.mulf %77, %78 : f64
// CHECK-NEXT:        %80 = stencil.access %arg9_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %81 = stencil.access %arg9_1[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %82 = arith.subf %80, %81 : f64
// CHECK-NEXT:        %83 = arith.addf %79, %82 : f64
// CHECK-NEXT:        %84 = stencil.access %arg10_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %85 = arith.divf %83, %84 : f64
// CHECK-NEXT:        %86 = stencil.store_result %85 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %86 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %42 to %6 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      stencil.store %76 to %5 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// SHAPE:         func.func @fvtp2d_qi(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>, %arg3 : !stencil.field<?x?x?xf64>, %arg4 : !stencil.field<?x?x?xf64>, %arg5 : !stencil.field<?x?x?xf64>, %arg6 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// SHAPE-NEXT:      %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      %2 = stencil.cast %arg2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      %3 = stencil.cast %arg3 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      %4 = stencil.cast %arg4 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      %5 = stencil.cast %arg5 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      %6 = stencil.cast %arg6 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      %7 = stencil.load %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:      %8 = stencil.load %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:      %9 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:      %10 = stencil.load %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:      %11 = stencil.load %4 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:      %12 = stencil.apply(%13 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[-1,66]x[0,64]xf64>) {
// SHAPE-NEXT:        %14 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:        %15 = arith.constant 7.000000e+00 : f64
// SHAPE-NEXT:        %16 = arith.constant 1.200000e+01 : f64
// SHAPE-NEXT:        %17 = arith.divf %15, %16 : f64
// SHAPE-NEXT:        %18 = arith.divf %14, %16 : f64
// SHAPE-NEXT:        %19 = stencil.access %13[0, -1, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %20 = stencil.access %13[0, 0, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %21 = arith.addf %19, %20 : f64
// SHAPE-NEXT:        %22 = stencil.access %13[0, -2, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %23 = stencil.access %13[0, 1, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %24 = arith.addf %22, %23 : f64
// SHAPE-NEXT:        %25 = arith.mulf %17, %21 : f64
// SHAPE-NEXT:        %26 = arith.mulf %18, %24 : f64
// SHAPE-NEXT:        %27 = arith.addf %25, %26 : f64
// SHAPE-NEXT:        %28 = stencil.store_result %27 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %28 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %29 = stencil.buffer %12 : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:      %30, %31, %32, %33 = stencil.apply(%34 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %35 = %29 : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>) {
// SHAPE-NEXT:        %36 = arith.constant 0.000000e+00 : f64
// SHAPE-NEXT:        %37 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:        %38 = stencil.access %35[0, 0, 0] : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:        %39 = stencil.access %34[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:        %40 = arith.subf %38, %39 : f64
// SHAPE-NEXT:        %41 = stencil.access %35[0, 1, 0] : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:        %42 = arith.subf %41, %39 : f64
// SHAPE-NEXT:        %43 = arith.addf %40, %42 : f64
// SHAPE-NEXT:        %44 = arith.mulf %40, %42 : f64
// SHAPE-NEXT:        %45 = arith.cmpf olt, %44, %36 : f64
// SHAPE-NEXT:        %46 = arith.select %45, %37, %36 : f64
// SHAPE-NEXT:        %47 = stencil.store_result %40 : !stencil.result<f64>
// SHAPE-NEXT:        %48 = stencil.store_result %42 : !stencil.result<f64>
// SHAPE-NEXT:        %49 = stencil.store_result %43 : !stencil.result<f64>
// SHAPE-NEXT:        %50 = stencil.store_result %46 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %47, %48, %49, %50 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %51 = stencil.buffer %30 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:      %52 = stencil.buffer %31 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:      %53 = stencil.buffer %32 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:      %54 = stencil.buffer %33 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:      %55 = stencil.apply(%arg7 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %arg8 = %8 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %arg9 = %51 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg10 = %52 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg11 = %53 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg12 = %54 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,65]x[0,64]xf64>) {
// SHAPE-NEXT:        %cst = arith.constant 0.000000e+00 : f64
// SHAPE-NEXT:        %cst_0 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:        %56 = stencil.access %arg12[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:        %57 = arith.cmpf oeq, %56, %cst : f64
// SHAPE-NEXT:        %58 = arith.select %57, %cst_0, %cst : f64
// SHAPE-NEXT:        %59 = stencil.access %arg12[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:        %60 = arith.mulf %59, %58 : f64
// SHAPE-NEXT:        %61 = arith.addf %56, %60 : f64
// SHAPE-NEXT:        %62 = stencil.access %arg8[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %63 = arith.cmpf ogt, %62, %cst : f64
// SHAPE-NEXT:        %64 = "scf.if"(%63) ({
// SHAPE-NEXT:          %65 = stencil.access %arg10[0, -1, 0] : !stencil.temp<[0,64]x[-1,64]x[0,64]xf64>
// SHAPE-NEXT:          %66 = stencil.access %arg11[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %67 = arith.mulf %62, %66 : f64
// SHAPE-NEXT:          %68 = arith.subf %65, %67 : f64
// SHAPE-NEXT:          %69 = arith.subf %cst_0, %62 : f64
// SHAPE-NEXT:          %70 = arith.mulf %69, %68 : f64
// SHAPE-NEXT:          scf.yield %70 : f64
// SHAPE-NEXT:        }, {
// SHAPE-NEXT:          %71 = stencil.access %arg9[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:          %72 = stencil.access %arg11[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %73 = arith.mulf %62, %72 : f64
// SHAPE-NEXT:          %74 = arith.addf %71, %73 : f64
// SHAPE-NEXT:          %75 = arith.addf %cst_0, %62 : f64
// SHAPE-NEXT:          %76 = arith.mulf %75, %74 : f64
// SHAPE-NEXT:          scf.yield %76 : f64
// SHAPE-NEXT:        }) : (i1) -> f64
// SHAPE-NEXT:        %77 = arith.mulf %64, %61 : f64
// SHAPE-NEXT:        %78 = "scf.if"(%63) ({
// SHAPE-NEXT:          %79 = stencil.access %arg7[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %80 = arith.addf %79, %77 : f64
// SHAPE-NEXT:          scf.yield %80 : f64
// SHAPE-NEXT:        }, {
// SHAPE-NEXT:          %81 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %82 = arith.addf %81, %77 : f64
// SHAPE-NEXT:          scf.yield %82 : f64
// SHAPE-NEXT:        }) : (i1) -> f64
// SHAPE-NEXT:        %83 = stencil.store_result %78 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %83 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %84 = stencil.apply(%85 = %10 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %86 = %55 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,65]x[0,64]xf64>) {
// SHAPE-NEXT:        %87 = stencil.access %85[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %88 = stencil.access %86[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %89 = arith.mulf %87, %88 : f64
// SHAPE-NEXT:        %90 = stencil.store_result %89 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %90 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %91 = stencil.buffer %84 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:      %92 = stencil.apply(%arg7_1 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %arg8_1 = %11 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>, %arg9_1 = %91 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %arg10_1 = %9 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// SHAPE-NEXT:        %93 = stencil.access %arg7_1[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:        %94 = stencil.access %arg8_1[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:        %95 = arith.mulf %93, %94 : f64
// SHAPE-NEXT:        %96 = stencil.access %arg9_1[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %97 = stencil.access %arg9_1[0, 1, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %98 = arith.subf %96, %97 : f64
// SHAPE-NEXT:        %99 = arith.addf %95, %98 : f64
// SHAPE-NEXT:        %100 = stencil.access %arg10_1[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:        %101 = arith.divf %99, %100 : f64
// SHAPE-NEXT:        %102 = stencil.store_result %101 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %102 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      stencil.store %55 to %6 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,65]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      stencil.store %92 to %5 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      func.return
// SHAPE-NEXT:    }
