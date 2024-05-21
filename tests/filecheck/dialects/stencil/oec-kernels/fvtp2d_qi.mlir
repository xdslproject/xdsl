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

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @fvtp2d_qi(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>, %arg3 : !stencil.field<?x?x?xf64>, %arg4 : !stencil.field<?x?x?xf64>, %arg5 : !stencil.field<?x?x?xf64>, %arg6 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:     %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %2 = stencil.cast %arg2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = stencil.cast %arg3 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = stencil.cast %arg4 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %5 = stencil.cast %arg5 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %6 = stencil.cast %arg6 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %7 = stencil.load %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:     %8 = stencil.load %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:     %9 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:     %10 = stencil.load %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:     %11 = stencil.load %4 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?x?xf64>
// CHECK-NEXT:     %12 = stencil.apply(%arg7 = %7 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:       %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:       %cst_1 = arith.constant 7.000000e+00 : f64
// CHECK-NEXT:       %cst_2 = arith.constant 1.200000e+01 : f64
// CHECK-NEXT:       %13 = arith.divf %cst_1, %cst_2 : f64
// CHECK-NEXT:       %14 = arith.divf %cst, %cst_2 : f64
// CHECK-NEXT:       %15 = stencil.access %arg7[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %16 = stencil.access %arg7[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %17 = arith.addf %15, %16 : f64
// CHECK-NEXT:       %18 = stencil.access %arg7[0, -2, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %19 = stencil.access %arg7[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %20 = arith.addf %18, %19 : f64
// CHECK-NEXT:       %21 = arith.mulf %13, %17 : f64
// CHECK-NEXT:       %22 = arith.mulf %14, %20 : f64
// CHECK-NEXT:       %23 = arith.addf %21, %22 : f64
// CHECK-NEXT:       %24 = stencil.store_result %23 : !stencil.result<f64>
// CHECK-NEXT:       stencil.return %24 : !stencil.result<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %13, %14, %15, %16 = stencil.apply(%arg7_1 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %12 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:       %cst_3 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:       %cst_4 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:       %17 = stencil.access %arg8[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %18 = stencil.access %arg7_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %19 = arith.subf %17, %18 : f64
// CHECK-NEXT:       %20 = stencil.access %arg8[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %21 = arith.subf %20, %18 : f64
// CHECK-NEXT:       %22 = arith.addf %19, %21 : f64
// CHECK-NEXT:       %23 = arith.mulf %19, %21 : f64
// CHECK-NEXT:       %24 = arith.cmpf olt, %23, %cst_3 : f64
// CHECK-NEXT:       %25 = arith.select %24, %cst_4, %cst_3 : f64
// CHECK-NEXT:       %26 = stencil.store_result %19 : !stencil.result<f64>
// CHECK-NEXT:       %27 = stencil.store_result %21 : !stencil.result<f64>
// CHECK-NEXT:       %28 = stencil.store_result %22 : !stencil.result<f64>
// CHECK-NEXT:       %29 = stencil.store_result %25 : !stencil.result<f64>
// CHECK-NEXT:       stencil.return %26, %27, %28, %29 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %17 = stencil.apply(%arg7_2 = %7 : !stencil.temp<?x?x?xf64>, %arg8_1 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %13 : !stencil.temp<?x?x?xf64>, %arg10 = %14 : !stencil.temp<?x?x?xf64>, %arg11 = %15 : !stencil.temp<?x?x?xf64>, %arg12 = %16 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:       %cst_5 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:       %cst_6 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:       %18 = stencil.access %arg12[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %19 = arith.cmpf oeq, %18, %cst_5 : f64
// CHECK-NEXT:       %20 = arith.select %19, %cst_6, %cst_5 : f64
// CHECK-NEXT:       %21 = stencil.access %arg12[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %22 = arith.mulf %21, %20 : f64
// CHECK-NEXT:       %23 = arith.addf %18, %22 : f64
// CHECK-NEXT:       %24 = stencil.access %arg8_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %25 = arith.cmpf ogt, %24, %cst_5 : f64
// CHECK-NEXT:       %26 = "scf.if"(%25) ({
// CHECK-NEXT:         %27 = stencil.access %arg10[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:         %28 = stencil.access %arg11[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:         %29 = arith.mulf %24, %28 : f64
// CHECK-NEXT:         %30 = arith.subf %27, %29 : f64
// CHECK-NEXT:         %31 = arith.subf %cst_6, %24 : f64
// CHECK-NEXT:         %32 = arith.mulf %31, %30 : f64
// CHECK-NEXT:         scf.yield %32 : f64
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %27 = stencil.access %arg9[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:         %28 = stencil.access %arg11[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:         %29 = arith.mulf %24, %28 : f64
// CHECK-NEXT:         %30 = arith.addf %27, %29 : f64
// CHECK-NEXT:         %31 = arith.addf %cst_6, %24 : f64
// CHECK-NEXT:         %32 = arith.mulf %31, %30 : f64
// CHECK-NEXT:         scf.yield %32 : f64
// CHECK-NEXT:       }) : (i1) -> f64
// CHECK-NEXT:       %27 = arith.mulf %26, %23 : f64
// CHECK-NEXT:       %28 = "scf.if"(%25) ({
// CHECK-NEXT:         %29 = stencil.access %arg7_2[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:         %30 = arith.addf %29, %27 : f64
// CHECK-NEXT:         scf.yield %30 : f64
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %29 = stencil.access %arg7_2[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:         %30 = arith.addf %29, %27 : f64
// CHECK-NEXT:         scf.yield %30 : f64
// CHECK-NEXT:       }) : (i1) -> f64
// CHECK-NEXT:       %29 = stencil.store_result %28 : !stencil.result<f64>
// CHECK-NEXT:       stencil.return %29 : !stencil.result<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %18 = stencil.apply(%arg7_3 = %10 : !stencil.temp<?x?x?xf64>, %arg8_2 = %17 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:       %19 = stencil.access %arg7_3[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %20 = stencil.access %arg8_2[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %21 = arith.mulf %19, %20 : f64
// CHECK-NEXT:       %22 = stencil.store_result %21 : !stencil.result<f64>
// CHECK-NEXT:       stencil.return %22 : !stencil.result<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %19 = stencil.apply(%arg7_4 = %7 : !stencil.temp<?x?x?xf64>, %arg8_3 = %11 : !stencil.temp<?x?x?xf64>, %arg9_1 = %18 : !stencil.temp<?x?x?xf64>, %arg10_1 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:       %20 = stencil.access %arg7_4[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %21 = stencil.access %arg8_3[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %22 = arith.mulf %20, %21 : f64
// CHECK-NEXT:       %23 = stencil.access %arg9_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %24 = stencil.access %arg9_1[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %25 = arith.subf %23, %24 : f64
// CHECK-NEXT:       %26 = arith.addf %22, %25 : f64
// CHECK-NEXT:       %27 = stencil.access %arg10_1[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:       %28 = arith.divf %26, %27 : f64
// CHECK-NEXT:       %29 = stencil.store_result %28 : !stencil.result<f64>
// CHECK-NEXT:       stencil.return %29 : !stencil.result<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     stencil.store %17 to %6 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     stencil.store %19 to %5 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// SHAPE-NEXT: builtin.module {
// SHAPE-NEXT:   func.func @fvtp2d_qi(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>, %arg3 : !stencil.field<?x?x?xf64>, %arg4 : !stencil.field<?x?x?xf64>, %arg5 : !stencil.field<?x?x?xf64>, %arg6 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// SHAPE-NEXT:     %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     %2 = stencil.cast %arg2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     %3 = stencil.cast %arg3 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     %4 = stencil.cast %arg4 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     %5 = stencil.cast %arg5 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     %6 = stencil.cast %arg6 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     %7 = stencil.load %0 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:     %8 = stencil.load %1 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:     %9 = stencil.load %2 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:     %10 = stencil.load %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:     %11 = stencil.load %4 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:     %12 = stencil.apply(%13 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[-1,66]x[0,64]xf64>) {
// SHAPE-NEXT:       %14 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:       %15 = arith.constant 7.000000e+00 : f64
// SHAPE-NEXT:       %16 = arith.constant 1.200000e+01 : f64
// SHAPE-NEXT:       %17 = arith.divf %15, %16 : f64
// SHAPE-NEXT:       %18 = arith.divf %14, %16 : f64
// SHAPE-NEXT:       %19 = stencil.access %13[0, -1, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:       %20 = stencil.access %13[0, 0, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:       %21 = arith.addf %19, %20 : f64
// SHAPE-NEXT:       %22 = stencil.access %13[0, -2, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:       %23 = stencil.access %13[0, 1, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:       %24 = arith.addf %22, %23 : f64
// SHAPE-NEXT:       %25 = arith.mulf %17, %21 : f64
// SHAPE-NEXT:       %26 = arith.mulf %18, %24 : f64
// SHAPE-NEXT:       %27 = arith.addf %25, %26 : f64
// SHAPE-NEXT:       %28 = stencil.store_result %27 : !stencil.result<f64>
// SHAPE-NEXT:       stencil.return %28 : !stencil.result<f64>
// SHAPE-NEXT:     }
// SHAPE-NEXT:     %14 = stencil.buffer %12 : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:     %15, %16, %17, %18 = stencil.apply(%19 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %20 = %14 : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>) {
// SHAPE-NEXT:       %21 = arith.constant 0.000000e+00 : f64
// SHAPE-NEXT:       %22 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:       %23 = stencil.access %20[0, 0, 0] : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:       %24 = stencil.access %19[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:       %25 = arith.subf %23, %24 : f64
// SHAPE-NEXT:       %26 = stencil.access %20[0, 1, 0] : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:       %27 = arith.subf %26, %24 : f64
// SHAPE-NEXT:       %28 = arith.addf %25, %27 : f64
// SHAPE-NEXT:       %29 = arith.mulf %25, %27 : f64
// SHAPE-NEXT:       %30 = arith.cmpf olt, %29, %21 : f64
// SHAPE-NEXT:       %31 = arith.select %30, %22, %21 : f64
// SHAPE-NEXT:       %32 = stencil.store_result %25 : !stencil.result<f64>
// SHAPE-NEXT:       %33 = stencil.store_result %27 : !stencil.result<f64>
// SHAPE-NEXT:       %34 = stencil.store_result %28 : !stencil.result<f64>
// SHAPE-NEXT:       %35 = stencil.store_result %31 : !stencil.result<f64>
// SHAPE-NEXT:       stencil.return %32, %33, %34, %35 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// SHAPE-NEXT:     }
// SHAPE-NEXT:     %21 = stencil.buffer %15 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:     %22 = stencil.buffer %16 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:     %23 = stencil.buffer %17 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:     %24 = stencil.buffer %18 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:     %25 = stencil.apply(%arg7 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %arg8 = %8 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %arg9 = %21 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg10 = %22 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg11 = %23 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg12 = %24 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,65]x[0,64]xf64>) {
// SHAPE-NEXT:       %cst = arith.constant 0.000000e+00 : f64
// SHAPE-NEXT:       %cst_1 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:       %26 = stencil.access %arg12[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:       %27 = arith.cmpf oeq, %26, %cst : f64
// SHAPE-NEXT:       %28 = arith.select %27, %cst_1, %cst : f64
// SHAPE-NEXT:       %29 = stencil.access %arg12[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:       %30 = arith.mulf %29, %28 : f64
// SHAPE-NEXT:       %31 = arith.addf %26, %30 : f64
// SHAPE-NEXT:       %32 = stencil.access %arg8[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:       %33 = arith.cmpf ogt, %32, %cst : f64
// SHAPE-NEXT:       %34 = "scf.if"(%33) ({
// SHAPE-NEXT:         %35 = stencil.access %arg10[0, -1, 0] : !stencil.temp<[0,64]x[-1,64]x[0,64]xf64>
// SHAPE-NEXT:         %36 = stencil.access %arg11[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:         %37 = arith.mulf %32, %36 : f64
// SHAPE-NEXT:         %38 = arith.subf %35, %37 : f64
// SHAPE-NEXT:         %39 = arith.subf %cst_1, %32 : f64
// SHAPE-NEXT:         %40 = arith.mulf %39, %38 : f64
// SHAPE-NEXT:         scf.yield %40 : f64
// SHAPE-NEXT:       }, {
// SHAPE-NEXT:         %35 = stencil.access %arg9[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:         %36 = stencil.access %arg11[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:         %37 = arith.mulf %32, %36 : f64
// SHAPE-NEXT:         %38 = arith.addf %35, %37 : f64
// SHAPE-NEXT:         %39 = arith.addf %cst_1, %32 : f64
// SHAPE-NEXT:         %40 = arith.mulf %39, %38 : f64
// SHAPE-NEXT:         scf.yield %40 : f64
// SHAPE-NEXT:       }) : (i1) -> f64
// SHAPE-NEXT:       %35 = arith.mulf %34, %31 : f64
// SHAPE-NEXT:       %36 = "scf.if"(%33) ({
// SHAPE-NEXT:         %37 = stencil.access %arg7[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:         %38 = arith.addf %37, %35 : f64
// SHAPE-NEXT:         scf.yield %38 : f64
// SHAPE-NEXT:       }, {
// SHAPE-NEXT:         %37 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:         %38 = arith.addf %37, %35 : f64
// SHAPE-NEXT:         scf.yield %38 : f64
// SHAPE-NEXT:       }) : (i1) -> f64
// SHAPE-NEXT:       %37 = stencil.store_result %36 : !stencil.result<f64>
// SHAPE-NEXT:       stencil.return %37 : !stencil.result<f64>
// SHAPE-NEXT:     }
// SHAPE-NEXT:     %26 = stencil.apply(%27 = %10 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %28 = %25 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,65]x[0,64]xf64>) {
// SHAPE-NEXT:       %29 = stencil.access %27[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:       %30 = stencil.access %28[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:       %31 = arith.mulf %29, %30 : f64
// SHAPE-NEXT:       %32 = stencil.store_result %31 : !stencil.result<f64>
// SHAPE-NEXT:       stencil.return %32 : !stencil.result<f64>
// SHAPE-NEXT:     }
// SHAPE-NEXT:     %29 = stencil.buffer %26 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:     %30 = stencil.apply(%arg7_1 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %arg8_1 = %11 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>, %arg9_1 = %29 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %arg10_1 = %9 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// SHAPE-NEXT:       %31 = stencil.access %arg7_1[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:       %32 = stencil.access %arg8_1[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:       %33 = arith.mulf %31, %32 : f64
// SHAPE-NEXT:       %34 = stencil.access %arg9_1[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:       %35 = stencil.access %arg9_1[0, 1, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:       %36 = arith.subf %34, %35 : f64
// SHAPE-NEXT:       %37 = arith.addf %33, %36 : f64
// SHAPE-NEXT:       %38 = stencil.access %arg10_1[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:       %39 = arith.divf %37, %38 : f64
// SHAPE-NEXT:       %40 = stencil.store_result %39 : !stencil.result<f64>
// SHAPE-NEXT:       stencil.return %40 : !stencil.result<f64>
// SHAPE-NEXT:     }
// SHAPE-NEXT:     stencil.store %25 to %6 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,65]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     stencil.store %30 to %5 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:     func.return
// SHAPE-NEXT:   }
// SHAPE-NEXT: }

// MLIR-NEXT: builtin.module {
// MLIR-NEXT:   func.func @fvtp2d_qi(%arg0 : memref<?x?x?xf64>, %arg1 : memref<?x?x?xf64>, %arg2 : memref<?x?x?xf64>, %arg3 : memref<?x?x?xf64>, %arg4 : memref<?x?x?xf64>, %arg5 : memref<?x?x?xf64>, %arg6 : memref<?x?x?xf64>)  attributes {"stencil.program"}{
// MLIR-NEXT:     %0 = memref.alloc() : memref<64x67x64xf64>
// MLIR-NEXT:     %1 = "memref.subview"(%0) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 67, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x67x64xf64>) -> memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:     %2 = memref.alloc() : memref<64x66x64xf64>
// MLIR-NEXT:     %arg9 = "memref.subview"(%2) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x66x64xf64>) -> memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:     %3 = memref.alloc() : memref<64x66x64xf64>
// MLIR-NEXT:     %arg10 = "memref.subview"(%3) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x66x64xf64>) -> memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:     %4 = memref.alloc() : memref<64x66x64xf64>
// MLIR-NEXT:     %arg11 = "memref.subview"(%4) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x66x64xf64>) -> memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:     %5 = memref.alloc() : memref<64x66x64xf64>
// MLIR-NEXT:     %arg12 = "memref.subview"(%5) <{"static_offsets" = array<i64: 0, 1, 0>, "static_sizes" = array<i64: 64, 66, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x66x64xf64>) -> memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:     %6 = memref.alloc() : memref<64x65x64xf64>
// MLIR-NEXT:     %arg9_1 = "memref.subview"(%6) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 64, 65, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<64x65x64xf64>) -> memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:     %7 = "memref.cast"(%arg0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:     %8 = "memref.cast"(%arg1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:     %9 = "memref.cast"(%arg2) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:     %10 = "memref.cast"(%arg3) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:     %11 = "memref.cast"(%arg4) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:     %12 = "memref.cast"(%arg5) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:     %13 = "memref.subview"(%12) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:     %14 = "memref.cast"(%arg6) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:     %15 = "memref.subview"(%14) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 65, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:     %16 = "memref.subview"(%7) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 70, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:     %17 = "memref.subview"(%8) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 65, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:     %18 = "memref.subview"(%9) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:     %19 = "memref.subview"(%10) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 65, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:     %20 = "memref.subview"(%11) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:     %21 = arith.constant 0 : index
// MLIR-NEXT:     %22 = arith.constant -1 : index
// MLIR-NEXT:     %23 = arith.constant 0 : index
// MLIR-NEXT:     %24 = arith.constant 1 : index
// MLIR-NEXT:     %25 = arith.constant 1 : index
// MLIR-NEXT:     %26 = arith.constant 1 : index
// MLIR-NEXT:     %27 = arith.constant 64 : index
// MLIR-NEXT:     %28 = arith.constant 66 : index
// MLIR-NEXT:     %29 = arith.constant 64 : index
// MLIR-NEXT:     "scf.parallel"(%21, %22, %23, %27, %28, %29, %24, %25, %26) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:     ^0(%30 : index, %31 : index, %32 : index):
// MLIR-NEXT:       %33 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:       %34 = arith.constant 7.000000e+00 : f64
// MLIR-NEXT:       %35 = arith.constant 1.200000e+01 : f64
// MLIR-NEXT:       %36 = arith.divf %34, %35 : f64
// MLIR-NEXT:       %37 = arith.divf %33, %35 : f64
// MLIR-NEXT:       %38 = arith.constant -1 : index
// MLIR-NEXT:       %39 = arith.addi %31, %38 : index
// MLIR-NEXT:       %40 = memref.load %16[%30, %39, %32] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %41 = memref.load %16[%30, %31, %32] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %42 = arith.addf %40, %41 : f64
// MLIR-NEXT:       %43 = arith.constant -2 : index
// MLIR-NEXT:       %44 = arith.addi %31, %43 : index
// MLIR-NEXT:       %45 = memref.load %16[%30, %44, %32] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %46 = arith.constant 1 : index
// MLIR-NEXT:       %47 = arith.addi %31, %46 : index
// MLIR-NEXT:       %48 = memref.load %16[%30, %47, %32] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %49 = arith.addf %45, %48 : f64
// MLIR-NEXT:       %50 = arith.mulf %36, %42 : f64
// MLIR-NEXT:       %51 = arith.mulf %37, %49 : f64
// MLIR-NEXT:       %52 = arith.addf %50, %51 : f64
// MLIR-NEXT:       memref.store %52, %1[%30, %31, %32] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:       scf.yield
// MLIR-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:     %30 = arith.constant 0 : index
// MLIR-NEXT:     %31 = arith.constant -1 : index
// MLIR-NEXT:     %32 = arith.constant 0 : index
// MLIR-NEXT:     %33 = arith.constant 1 : index
// MLIR-NEXT:     %34 = arith.constant 1 : index
// MLIR-NEXT:     %35 = arith.constant 1 : index
// MLIR-NEXT:     %36 = arith.constant 64 : index
// MLIR-NEXT:     %37 = arith.constant 65 : index
// MLIR-NEXT:     %38 = arith.constant 64 : index
// MLIR-NEXT:     "scf.parallel"(%30, %31, %32, %36, %37, %38, %33, %34, %35) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:     ^0(%39 : index, %40 : index, %41 : index):
// MLIR-NEXT:       %42 = arith.constant 0.000000e+00 : f64
// MLIR-NEXT:       %43 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:       %44 = memref.load %1[%39, %40, %41] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:       %45 = memref.load %16[%39, %40, %41] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %46 = arith.subf %44, %45 : f64
// MLIR-NEXT:       %47 = arith.constant 1 : index
// MLIR-NEXT:       %48 = arith.addi %40, %47 : index
// MLIR-NEXT:       %49 = memref.load %1[%39, %48, %41] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:       %50 = arith.subf %49, %45 : f64
// MLIR-NEXT:       %51 = arith.addf %46, %50 : f64
// MLIR-NEXT:       %52 = arith.mulf %46, %50 : f64
// MLIR-NEXT:       %53 = arith.cmpf olt, %52, %42 : f64
// MLIR-NEXT:       %54 = arith.select %53, %43, %42 : f64
// MLIR-NEXT:       memref.store %46, %arg9[%39, %40, %41] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:       memref.store %50, %arg10[%39, %40, %41] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:       memref.store %51, %arg11[%39, %40, %41] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:       memref.store %54, %arg12[%39, %40, %41] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:       scf.yield
// MLIR-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:     %39 = arith.constant 0 : index
// MLIR-NEXT:     %40 = arith.constant 0 : index
// MLIR-NEXT:     %41 = arith.constant 0 : index
// MLIR-NEXT:     %42 = arith.constant 1 : index
// MLIR-NEXT:     %43 = arith.constant 1 : index
// MLIR-NEXT:     %44 = arith.constant 1 : index
// MLIR-NEXT:     %45 = arith.constant 64 : index
// MLIR-NEXT:     %46 = arith.constant 65 : index
// MLIR-NEXT:     %47 = arith.constant 64 : index
// MLIR-NEXT:     "scf.parallel"(%39, %40, %41, %45, %46, %47, %42, %43, %44) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:     ^0(%48 : index, %49 : index, %50 : index):
// MLIR-NEXT:       %cst = arith.constant 0.000000e+00 : f64
// MLIR-NEXT:       %cst_1 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:       %51 = arith.constant -1 : index
// MLIR-NEXT:       %52 = arith.addi %49, %51 : index
// MLIR-NEXT:       %53 = memref.load %arg12[%48, %52, %50] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:       %54 = arith.cmpf oeq, %53, %cst : f64
// MLIR-NEXT:       %55 = arith.select %54, %cst_1, %cst : f64
// MLIR-NEXT:       %56 = memref.load %arg12[%48, %49, %50] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:       %57 = arith.mulf %56, %55 : f64
// MLIR-NEXT:       %58 = arith.addf %53, %57 : f64
// MLIR-NEXT:       %59 = memref.load %17[%48, %49, %50] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %60 = arith.cmpf ogt, %59, %cst : f64
// MLIR-NEXT:       %61 = "scf.if"(%60) ({
// MLIR-NEXT:         %62 = arith.constant -1 : index
// MLIR-NEXT:         %63 = arith.addi %49, %62 : index
// MLIR-NEXT:         %64 = memref.load %arg10[%48, %63, %50] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:         %65 = arith.constant -1 : index
// MLIR-NEXT:         %66 = arith.addi %49, %65 : index
// MLIR-NEXT:         %67 = memref.load %arg11[%48, %66, %50] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:         %68 = arith.mulf %59, %67 : f64
// MLIR-NEXT:         %69 = arith.subf %64, %68 : f64
// MLIR-NEXT:         %70 = arith.subf %cst_1, %59 : f64
// MLIR-NEXT:         %71 = arith.mulf %70, %69 : f64
// MLIR-NEXT:         scf.yield %71 : f64
// MLIR-NEXT:       }, {
// MLIR-NEXT:         %62 = memref.load %arg9[%48, %49, %50] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:         %63 = memref.load %arg11[%48, %49, %50] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:         %64 = arith.mulf %59, %63 : f64
// MLIR-NEXT:         %65 = arith.addf %62, %64 : f64
// MLIR-NEXT:         %66 = arith.addf %cst_1, %59 : f64
// MLIR-NEXT:         %67 = arith.mulf %66, %65 : f64
// MLIR-NEXT:         scf.yield %67 : f64
// MLIR-NEXT:       }) : (i1) -> f64
// MLIR-NEXT:       %62 = arith.mulf %61, %58 : f64
// MLIR-NEXT:       %63 = "scf.if"(%60) ({
// MLIR-NEXT:         %64 = arith.constant -1 : index
// MLIR-NEXT:         %65 = arith.addi %49, %64 : index
// MLIR-NEXT:         %66 = memref.load %16[%48, %65, %50] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:         %67 = arith.addf %66, %62 : f64
// MLIR-NEXT:         scf.yield %67 : f64
// MLIR-NEXT:       }, {
// MLIR-NEXT:         %64 = memref.load %16[%48, %49, %50] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:         %65 = arith.addf %64, %62 : f64
// MLIR-NEXT:         scf.yield %65 : f64
// MLIR-NEXT:       }) : (i1) -> f64
// MLIR-NEXT:       memref.store %63, %15[%48, %49, %50] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       scf.yield
// MLIR-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:     %48 = arith.constant 0 : index
// MLIR-NEXT:     %49 = arith.constant 0 : index
// MLIR-NEXT:     %50 = arith.constant 0 : index
// MLIR-NEXT:     %51 = arith.constant 1 : index
// MLIR-NEXT:     %52 = arith.constant 1 : index
// MLIR-NEXT:     %53 = arith.constant 1 : index
// MLIR-NEXT:     %54 = arith.constant 64 : index
// MLIR-NEXT:     %55 = arith.constant 65 : index
// MLIR-NEXT:     %56 = arith.constant 64 : index
// MLIR-NEXT:     "scf.parallel"(%48, %49, %50, %54, %55, %56, %51, %52, %53) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:     ^0(%57 : index, %58 : index, %59 : index):
// MLIR-NEXT:       %60 = memref.load %19[%57, %58, %59] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %61 = memref.load %15[%57, %58, %59] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %62 = arith.mulf %60, %61 : f64
// MLIR-NEXT:       memref.store %62, %arg9_1[%57, %58, %59] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:       scf.yield
// MLIR-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:     %57 = arith.constant 0 : index
// MLIR-NEXT:     %58 = arith.constant 0 : index
// MLIR-NEXT:     %59 = arith.constant 0 : index
// MLIR-NEXT:     %60 = arith.constant 1 : index
// MLIR-NEXT:     %61 = arith.constant 1 : index
// MLIR-NEXT:     %62 = arith.constant 1 : index
// MLIR-NEXT:     %63 = arith.constant 64 : index
// MLIR-NEXT:     %64 = arith.constant 64 : index
// MLIR-NEXT:     %65 = arith.constant 64 : index
// MLIR-NEXT:     "scf.parallel"(%57, %58, %59, %63, %64, %65, %60, %61, %62) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:     ^0(%66 : index, %67 : index, %68 : index):
// MLIR-NEXT:       %69 = memref.load %16[%66, %67, %68] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %70 = memref.load %20[%66, %67, %68] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %71 = arith.mulf %69, %70 : f64
// MLIR-NEXT:       %72 = memref.load %arg9_1[%66, %67, %68] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:       %73 = arith.constant 1 : index
// MLIR-NEXT:       %74 = arith.addi %67, %73 : index
// MLIR-NEXT:       %75 = memref.load %arg9_1[%66, %74, %68] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:       %76 = arith.subf %72, %75 : f64
// MLIR-NEXT:       %77 = arith.addf %71, %76 : f64
// MLIR-NEXT:       %78 = memref.load %18[%66, %67, %68] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       %79 = arith.divf %77, %78 : f64
// MLIR-NEXT:       memref.store %79, %13[%66, %67, %68] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:       scf.yield
// MLIR-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:     memref.dealloc %6 : memref<64x65x64xf64>
// MLIR-NEXT:     memref.dealloc %5 : memref<64x66x64xf64>
// MLIR-NEXT:     memref.dealloc %4 : memref<64x66x64xf64>
// MLIR-NEXT:     memref.dealloc %3 : memref<64x66x64xf64>
// MLIR-NEXT:     memref.dealloc %2 : memref<64x66x64xf64>
// MLIR-NEXT:     memref.dealloc %0 : memref<64x67x64xf64>
// MLIR-NEXT:     func.return
// MLIR-NEXT:   }
// MLIR-NEXT: }
