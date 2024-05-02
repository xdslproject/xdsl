// RUN: XDSL_ROUNDTRIP
// RUN: xdsl-opt %s -p stencil-storage-materialization,stencil-shape-inference | filecheck %s --check-prefix SHAPE
// RUN: xdsl-opt %s -p stencil-storage-materialization,stencil-shape-inference,convert-stencil-to-ll-mlir | filecheck %s --check-prefix MLIR

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
// CHECK-NEXT:        %cst_1 = arith.constant 7.000000e+00 : f64
// CHECK-NEXT:        %cst_2 = arith.constant 1.200000e+01 : f64
// CHECK-NEXT:        %13 = arith.divf %cst_1, %cst_2 : f64
// CHECK-NEXT:        %14 = arith.divf %cst, %cst_2 : f64
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
// CHECK-NEXT:        %cst_3 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:        %cst_4 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %29 = stencil.access %arg8[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %30 = stencil.access %arg7_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %31 = arith.subf %29, %30 : f64
// CHECK-NEXT:        %32 = stencil.access %arg8[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %33 = arith.subf %32, %30 : f64
// CHECK-NEXT:        %34 = arith.addf %31, %33 : f64
// CHECK-NEXT:        %35 = arith.mulf %31, %33 : f64
// CHECK-NEXT:        %36 = arith.cmpf olt, %35, %cst_3 : f64
// CHECK-NEXT:        %37 = arith.select %36, %cst_4, %cst_3 : f64
// CHECK-NEXT:        %38 = stencil.store_result %31 : !stencil.result<f64>
// CHECK-NEXT:        %39 = stencil.store_result %33 : !stencil.result<f64>
// CHECK-NEXT:        %40 = stencil.store_result %34 : !stencil.result<f64>
// CHECK-NEXT:        %41 = stencil.store_result %37 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %38, %39, %40, %41 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %42 = stencil.apply(%arg7_2 = %7 : !stencil.temp<?x?x?xf64>, %arg8_1 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %25 : !stencil.temp<?x?x?xf64>, %arg10 = %26 : !stencil.temp<?x?x?xf64>, %arg11 = %27 : !stencil.temp<?x?x?xf64>, %arg12 = %28 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %cst_5 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:        %cst_6 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %43 = stencil.access %arg12[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %44 = arith.cmpf oeq, %43, %cst_5 : f64
// CHECK-NEXT:        %45 = arith.select %44, %cst_6, %cst_5 : f64
// CHECK-NEXT:        %46 = stencil.access %arg12[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %47 = arith.mulf %46, %45 : f64
// CHECK-NEXT:        %48 = arith.addf %43, %47 : f64
// CHECK-NEXT:        %49 = stencil.access %arg8_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %50 = arith.cmpf ogt, %49, %cst_5 : f64
// CHECK-NEXT:        %51 = "scf.if"(%50) ({
// CHECK-NEXT:          %52 = stencil.access %arg10[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %53 = stencil.access %arg11[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %54 = arith.mulf %49, %53 : f64
// CHECK-NEXT:          %55 = arith.subf %52, %54 : f64
// CHECK-NEXT:          %56 = arith.subf %cst_6, %49 : f64
// CHECK-NEXT:          %57 = arith.mulf %56, %55 : f64
// CHECK-NEXT:          scf.yield %57 : f64
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %58 = stencil.access %arg9[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %59 = stencil.access %arg11[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %60 = arith.mulf %49, %59 : f64
// CHECK-NEXT:          %61 = arith.addf %58, %60 : f64
// CHECK-NEXT:          %62 = arith.addf %cst_6, %49 : f64
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
// SHAPE-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:        %56 = stencil.access %arg12[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:        %57 = arith.cmpf oeq, %56, %cst : f64
// SHAPE-NEXT:        %58 = arith.select %57, %cst_1, %cst : f64
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
// SHAPE-NEXT:          %69 = arith.subf %cst_1, %62 : f64
// SHAPE-NEXT:          %70 = arith.mulf %69, %68 : f64
// SHAPE-NEXT:          scf.yield %70 : f64
// SHAPE-NEXT:        }, {
// SHAPE-NEXT:          %71 = stencil.access %arg9[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:          %72 = stencil.access %arg11[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %73 = arith.mulf %62, %72 : f64
// SHAPE-NEXT:          %74 = arith.addf %71, %73 : f64
// SHAPE-NEXT:          %75 = arith.addf %cst_1, %62 : f64
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

// MLIR:         func.func @fvtp2d_qi(%arg0 : memref<?x?x?xf64>, %arg1 : memref<?x?x?xf64>, %arg2 : memref<?x?x?xf64>, %arg3 : memref<?x?x?xf64>, %arg4 : memref<?x?x?xf64>, %arg5 : memref<?x?x?xf64>, %arg6 : memref<?x?x?xf64>)  attributes {"stencil.program"}{
// MLIR-NEXT:      %0 = memref.alloc() : memref<64x67x64xf64>
// MLIR-NEXT:      %1 = "memref.subview"(%0) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 67, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x67x64xf64>) -> memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:      %2 = memref.alloc() : memref<64x66x64xf64>
// MLIR-NEXT:      %arg9 = "memref.subview"(%2) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x66x64xf64>) -> memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      %3 = memref.alloc() : memref<64x66x64xf64>
// MLIR-NEXT:      %arg10 = "memref.subview"(%3) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x66x64xf64>) -> memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      %4 = memref.alloc() : memref<64x66x64xf64>
// MLIR-NEXT:      %arg11 = "memref.subview"(%4) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x66x64xf64>) -> memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      %5 = memref.alloc() : memref<64x66x64xf64>
// MLIR-NEXT:      %arg12 = "memref.subview"(%5) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x66x64xf64>) -> memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      %6 = memref.alloc() : memref<64x65x64xf64>
// MLIR-NEXT:      %arg9_1 = "memref.subview"(%6) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 64, 65, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x65x64xf64>) -> memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:      %7 = "memref.cast"(%arg0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %8 = "memref.cast"(%arg1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %9 = "memref.cast"(%arg2) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %10 = "memref.cast"(%arg3) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %11 = "memref.cast"(%arg4) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %12 = "memref.cast"(%arg5) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %13 = "memref.subview"(%12) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %14 = "memref.cast"(%arg6) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %15 = "memref.subview"(%14) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 65, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %16 = "memref.subview"(%7) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 70, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %17 = "memref.subview"(%8) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 65, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %18 = "memref.subview"(%9) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %19 = "memref.subview"(%10) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 65, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %20 = "memref.subview"(%11) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %21 = arith.constant 0 : index
// MLIR-NEXT:      %22 = arith.constant -1 : index
// MLIR-NEXT:      %23 = arith.constant 0 : index
// MLIR-NEXT:      %24 = arith.constant 1 : index
// MLIR-NEXT:      %25 = arith.constant 1 : index
// MLIR-NEXT:      %26 = arith.constant 1 : index
// MLIR-NEXT:      %27 = arith.constant 64 : index
// MLIR-NEXT:      %28 = arith.constant 66 : index
// MLIR-NEXT:      %29 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%21, %22, %23, %27, %28, %29, %24, %25, %26) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^0(%30 : index, %31 : index, %32 : index):
// MLIR-NEXT:        %33 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:        %34 = arith.constant 7.000000e+00 : f64
// MLIR-NEXT:        %35 = arith.constant 1.200000e+01 : f64
// MLIR-NEXT:        %36 = arith.divf %34, %35 : f64
// MLIR-NEXT:        %37 = arith.divf %33, %35 : f64
// MLIR-NEXT:        %38 = arith.constant -1 : index
// MLIR-NEXT:        %39 = arith.addi %31, %38 : index
// MLIR-NEXT:        %40 = memref.load %16[%30, %39, %32] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %41 = memref.load %16[%30, %31, %32] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %42 = arith.addf %40, %41 : f64
// MLIR-NEXT:        %43 = arith.constant -2 : index
// MLIR-NEXT:        %44 = arith.addi %31, %43 : index
// MLIR-NEXT:        %45 = memref.load %16[%30, %44, %32] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %46 = arith.constant 1 : index
// MLIR-NEXT:        %47 = arith.addi %31, %46 : index
// MLIR-NEXT:        %48 = memref.load %16[%30, %47, %32] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %49 = arith.addf %45, %48 : f64
// MLIR-NEXT:        %50 = arith.mulf %36, %42 : f64
// MLIR-NEXT:        %51 = arith.mulf %37, %49 : f64
// MLIR-NEXT:        %52 = arith.addf %50, %51 : f64
// MLIR-NEXT:        memref.store %52, %1[%30, %31, %32] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:        scf.yield
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      %53 = arith.constant 0 : index
// MLIR-NEXT:      %54 = arith.constant -1 : index
// MLIR-NEXT:      %55 = arith.constant 0 : index
// MLIR-NEXT:      %56 = arith.constant 1 : index
// MLIR-NEXT:      %57 = arith.constant 1 : index
// MLIR-NEXT:      %58 = arith.constant 1 : index
// MLIR-NEXT:      %59 = arith.constant 64 : index
// MLIR-NEXT:      %60 = arith.constant 65 : index
// MLIR-NEXT:      %61 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%53, %54, %55, %59, %60, %61, %56, %57, %58) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^1(%62 : index, %63 : index, %64 : index):
// MLIR-NEXT:        %65 = arith.constant 0.000000e+00 : f64
// MLIR-NEXT:        %66 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:        %67 = memref.load %1[%62, %63, %64] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:        %68 = memref.load %16[%62, %63, %64] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %69 = arith.subf %67, %68 : f64
// MLIR-NEXT:        %70 = arith.constant 1 : index
// MLIR-NEXT:        %71 = arith.addi %63, %70 : index
// MLIR-NEXT:        %72 = memref.load %1[%62, %71, %64] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:        %73 = arith.subf %72, %68 : f64
// MLIR-NEXT:        %74 = arith.addf %69, %73 : f64
// MLIR-NEXT:        %75 = arith.mulf %69, %73 : f64
// MLIR-NEXT:        %76 = arith.cmpf olt, %75, %65 : f64
// MLIR-NEXT:        %77 = arith.select %76, %66, %65 : f64
// MLIR-NEXT:        memref.store %69, %arg9[%62, %63, %64] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        memref.store %73, %arg10[%62, %63, %64] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        memref.store %74, %arg11[%62, %63, %64] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        memref.store %77, %arg12[%62, %63, %64] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        scf.yield
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      %78 = arith.constant 0 : index
// MLIR-NEXT:      %79 = arith.constant 0 : index
// MLIR-NEXT:      %80 = arith.constant 0 : index
// MLIR-NEXT:      %81 = arith.constant 1 : index
// MLIR-NEXT:      %82 = arith.constant 1 : index
// MLIR-NEXT:      %83 = arith.constant 1 : index
// MLIR-NEXT:      %84 = arith.constant 64 : index
// MLIR-NEXT:      %85 = arith.constant 65 : index
// MLIR-NEXT:      %86 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%78, %79, %80, %84, %85, %86, %81, %82, %83) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^2(%87 : index, %88 : index, %89 : index):
// MLIR-NEXT:        %cst = arith.constant 0.000000e+00 : f64
// MLIR-NEXT:        %cst_0 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:        %90 = arith.constant -1 : index
// MLIR-NEXT:        %91 = arith.addi %88, %90 : index
// MLIR-NEXT:        %92 = memref.load %arg12[%87, %91, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        %93 = arith.cmpf oeq, %92, %cst : f64
// MLIR-NEXT:        %94 = arith.select %93, %cst_0, %cst : f64
// MLIR-NEXT:        %95 = memref.load %arg12[%87, %88, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        %96 = arith.mulf %95, %94 : f64
// MLIR-NEXT:        %97 = arith.addf %92, %96 : f64
// MLIR-NEXT:        %98 = memref.load %17[%87, %88, %89] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %99 = arith.cmpf ogt, %98, %cst : f64
// MLIR-NEXT:        %100 = "scf.if"(%99) ({
// MLIR-NEXT:          %101 = arith.constant -1 : index
// MLIR-NEXT:          %102 = arith.addi %88, %101 : index
// MLIR-NEXT:          %103 = memref.load %arg10[%87, %102, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %104 = arith.constant -1 : index
// MLIR-NEXT:          %105 = arith.addi %88, %104 : index
// MLIR-NEXT:          %106 = memref.load %arg11[%87, %105, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %107 = arith.mulf %98, %106 : f64
// MLIR-NEXT:          %108 = arith.subf %103, %107 : f64
// MLIR-NEXT:          %109 = arith.subf %cst_0, %98 : f64
// MLIR-NEXT:          %110 = arith.mulf %109, %108 : f64
// MLIR-NEXT:          scf.yield %110 : f64
// MLIR-NEXT:        }, {
// MLIR-NEXT:          %111 = memref.load %arg9[%87, %88, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %112 = memref.load %arg11[%87, %88, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %113 = arith.mulf %98, %112 : f64
// MLIR-NEXT:          %114 = arith.addf %111, %113 : f64
// MLIR-NEXT:          %115 = arith.addf %cst_0, %98 : f64
// MLIR-NEXT:          %116 = arith.mulf %115, %114 : f64
// MLIR-NEXT:          scf.yield %116 : f64
// MLIR-NEXT:        }) : (i1) -> f64
// MLIR-NEXT:        %117 = arith.mulf %100, %97 : f64
// MLIR-NEXT:        %118 = "scf.if"(%99) ({
// MLIR-NEXT:          %119 = arith.constant -1 : index
// MLIR-NEXT:          %120 = arith.addi %88, %119 : index
// MLIR-NEXT:          %121 = memref.load %16[%87, %120, %89] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:          %122 = arith.addf %121, %117 : f64
// MLIR-NEXT:          scf.yield %122 : f64
// MLIR-NEXT:        }, {
// MLIR-NEXT:          %123 = memref.load %16[%87, %88, %89] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:          %124 = arith.addf %123, %117 : f64
// MLIR-NEXT:          scf.yield %124 : f64
// MLIR-NEXT:        }) : (i1) -> f64
// MLIR-NEXT:        memref.store %118, %15[%87, %88, %89] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        scf.yield
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      %125 = arith.constant 0 : index
// MLIR-NEXT:      %126 = arith.constant 0 : index
// MLIR-NEXT:      %127 = arith.constant 0 : index
// MLIR-NEXT:      %128 = arith.constant 1 : index
// MLIR-NEXT:      %129 = arith.constant 1 : index
// MLIR-NEXT:      %130 = arith.constant 1 : index
// MLIR-NEXT:      %131 = arith.constant 64 : index
// MLIR-NEXT:      %132 = arith.constant 65 : index
// MLIR-NEXT:      %133 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%125, %126, %127, %131, %132, %133, %128, %129, %130) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^3(%134 : index, %135 : index, %136 : index):
// MLIR-NEXT:        %137 = memref.load %19[%134, %135, %136] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %138 = memref.load %15[%134, %135, %136] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %139 = arith.mulf %137, %138 : f64
// MLIR-NEXT:        memref.store %139, %arg9_1[%134, %135, %136] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:        scf.yield
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      %140 = arith.constant 0 : index
// MLIR-NEXT:      %141 = arith.constant 0 : index
// MLIR-NEXT:      %142 = arith.constant 0 : index
// MLIR-NEXT:      %143 = arith.constant 1 : index
// MLIR-NEXT:      %144 = arith.constant 1 : index
// MLIR-NEXT:      %145 = arith.constant 1 : index
// MLIR-NEXT:      %146 = arith.constant 64 : index
// MLIR-NEXT:      %147 = arith.constant 64 : index
// MLIR-NEXT:      %148 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%140, %141, %142, %146, %147, %148, %143, %144, %145) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^4(%149 : index, %150 : index, %151 : index):
// MLIR-NEXT:        %152 = memref.load %16[%149, %150, %151] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %153 = memref.load %20[%149, %150, %151] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %154 = arith.mulf %152, %153 : f64
// MLIR-NEXT:        %155 = memref.load %arg9_1[%149, %150, %151] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:        %156 = arith.constant 1 : index
// MLIR-NEXT:        %157 = arith.addi %150, %156 : index
// MLIR-NEXT:        %158 = memref.load %arg9_1[%149, %157, %151] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:        %159 = arith.subf %155, %158 : f64
// MLIR-NEXT:        %160 = arith.addf %154, %159 : f64
// MLIR-NEXT:        %161 = memref.load %18[%149, %150, %151] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %162 = arith.divf %160, %161 : f64
// MLIR-NEXT:        memref.store %162, %13[%149, %150, %151] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        scf.yield
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      memref.dealloc %6 : memref<64x65x64xf64>
// MLIR-NEXT:      memref.dealloc %5 : memref<64x66x64xf64>
// MLIR-NEXT:      memref.dealloc %4 : memref<64x66x64xf64>
// MLIR-NEXT:      memref.dealloc %3 : memref<64x66x64xf64>
// MLIR-NEXT:      memref.dealloc %2 : memref<64x66x64xf64>
// MLIR-NEXT:      memref.dealloc %0 : memref<64x67x64xf64>
// MLIR-NEXT:      func.return
// MLIR-NEXT:    }
