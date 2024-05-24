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

// CHECK:    func.func @fvtp2d_qi(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>, %arg3 : !stencil.field<?x?x?xf64>, %arg4 : !stencil.field<?x?x?xf64>, %arg5 : !stencil.field<?x?x?xf64>, %arg6 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
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
// CHECK-NEXT:      %13, %14, %15, %16 = stencil.apply(%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %12 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %17 = stencil.access %arg8[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %18 = stencil.access %arg7[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %19 = arith.subf %17, %18 : f64
// CHECK-NEXT:        %20 = stencil.access %arg8[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %21 = arith.subf %20, %18 : f64
// CHECK-NEXT:        %22 = arith.addf %19, %21 : f64
// CHECK-NEXT:        %23 = arith.mulf %19, %21 : f64
// CHECK-NEXT:        %24 = arith.cmpf olt, %23, %cst : f64
// CHECK-NEXT:        %25 = arith.select %24, %cst_1, %cst : f64
// CHECK-NEXT:        %26 = stencil.store_result %19 : !stencil.result<f64>
// CHECK-NEXT:        %27 = stencil.store_result %21 : !stencil.result<f64>
// CHECK-NEXT:        %28 = stencil.store_result %22 : !stencil.result<f64>
// CHECK-NEXT:        %29 = stencil.store_result %25 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %26, %27, %28, %29 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %17 = stencil.apply(%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %13 : !stencil.temp<?x?x?xf64>, %arg10 = %14 : !stencil.temp<?x?x?xf64>, %arg11 = %15 : !stencil.temp<?x?x?xf64>, %arg12 = %16 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %18 = stencil.access %arg12[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %19 = arith.cmpf oeq, %18, %cst : f64
// CHECK-NEXT:        %20 = arith.select %19, %cst_1, %cst : f64
// CHECK-NEXT:        %21 = stencil.access %arg12[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %22 = arith.mulf %21, %20 : f64
// CHECK-NEXT:        %23 = arith.addf %18, %22 : f64
// CHECK-NEXT:        %24 = stencil.access %arg8[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %25 = arith.cmpf ogt, %24, %cst : f64
// CHECK-NEXT:        %26 = "scf.if"(%25) ({
// CHECK-NEXT:          %27 = stencil.access %arg10[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %28 = stencil.access %arg11[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %29 = arith.mulf %24, %28 : f64
// CHECK-NEXT:          %30 = arith.subf %27, %29 : f64
// CHECK-NEXT:          %31 = arith.subf %cst_1, %24 : f64
// CHECK-NEXT:          %32 = arith.mulf %31, %30 : f64
// CHECK-NEXT:          scf.yield %32 : f64
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %33 = stencil.access %arg9[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %34 = stencil.access %arg11[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %35 = arith.mulf %24, %34 : f64
// CHECK-NEXT:          %36 = arith.addf %33, %35 : f64
// CHECK-NEXT:          %37 = arith.addf %cst_1, %24 : f64
// CHECK-NEXT:          %38 = arith.mulf %37, %36 : f64
// CHECK-NEXT:          scf.yield %38 : f64
// CHECK-NEXT:        }) : (i1) -> f64
// CHECK-NEXT:        %39 = arith.mulf %26, %23 : f64
// CHECK-NEXT:        %40 = "scf.if"(%25) ({
// CHECK-NEXT:          %41 = stencil.access %arg7[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %42 = arith.addf %41, %39 : f64
// CHECK-NEXT:          scf.yield %42 : f64
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %43 = stencil.access %arg7[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %44 = arith.addf %43, %39 : f64
// CHECK-NEXT:          scf.yield %44 : f64
// CHECK-NEXT:        }) : (i1) -> f64
// CHECK-NEXT:        %45 = stencil.store_result %40 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %45 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %18 = stencil.apply(%arg7 = %10 : !stencil.temp<?x?x?xf64>, %arg8 = %17 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %19 = stencil.access %arg7[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %20 = stencil.access %arg8[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %21 = arith.mulf %19, %20 : f64
// CHECK-NEXT:        %22 = stencil.store_result %21 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %22 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %19 = stencil.apply(%arg7 = %7 : !stencil.temp<?x?x?xf64>, %arg8 = %11 : !stencil.temp<?x?x?xf64>, %arg9 = %18 : !stencil.temp<?x?x?xf64>, %arg10 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>) {
// CHECK-NEXT:        %20 = stencil.access %arg7[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %21 = stencil.access %arg8[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %22 = arith.mulf %20, %21 : f64
// CHECK-NEXT:        %23 = stencil.access %arg9[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %24 = stencil.access %arg9[0, 1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %25 = arith.subf %23, %24 : f64
// CHECK-NEXT:        %26 = arith.addf %22, %25 : f64
// CHECK-NEXT:        %27 = stencil.access %arg10[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:        %28 = arith.divf %26, %27 : f64
// CHECK-NEXT:        %29 = stencil.store_result %28 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %29 : !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %17 to %6 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      stencil.store %19 to %5 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
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
// SHAPE-NEXT:      %13 = stencil.buffer %12 : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:      %14, %15, %16, %17 = stencil.apply(%18 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %19 = %13 : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>) {
// SHAPE-NEXT:        %20 = arith.constant 0.000000e+00 : f64
// SHAPE-NEXT:        %21 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:        %22 = stencil.access %19[0, 0, 0] : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:        %23 = stencil.access %18[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:        %24 = arith.subf %22, %23 : f64
// SHAPE-NEXT:        %25 = stencil.access %19[0, 1, 0] : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:        %26 = arith.subf %25, %23 : f64
// SHAPE-NEXT:        %27 = arith.addf %24, %26 : f64
// SHAPE-NEXT:        %28 = arith.mulf %24, %26 : f64
// SHAPE-NEXT:        %29 = arith.cmpf olt, %28, %20 : f64
// SHAPE-NEXT:        %30 = arith.select %29, %21, %20 : f64
// SHAPE-NEXT:        %31 = stencil.store_result %24 : !stencil.result<f64>
// SHAPE-NEXT:        %32 = stencil.store_result %26 : !stencil.result<f64>
// SHAPE-NEXT:        %33 = stencil.store_result %27 : !stencil.result<f64>
// SHAPE-NEXT:        %34 = stencil.store_result %30 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %31, %32, %33, %34 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %18 = stencil.buffer %14 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:      %19 = stencil.buffer %15 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:      %20 = stencil.buffer %16 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:      %21 = stencil.buffer %17 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:      %22 = stencil.apply(%arg7 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %arg8 = %8 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %arg9 = %18 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg10 = %19 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg11 = %20 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, %arg12 = %21 : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,65]x[0,64]xf64>) {
// SHAPE-NEXT:        %cst = arith.constant 0.000000e+00 : f64
// SHAPE-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:        %23 = stencil.access %arg12[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:        %24 = arith.cmpf oeq, %23, %cst : f64
// SHAPE-NEXT:        %25 = arith.select %24, %cst_1, %cst : f64
// SHAPE-NEXT:        %26 = stencil.access %arg12[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:        %27 = arith.mulf %26, %25 : f64
// SHAPE-NEXT:        %28 = arith.addf %23, %27 : f64
// SHAPE-NEXT:        %29 = stencil.access %arg8[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %30 = arith.cmpf ogt, %29, %cst : f64
// SHAPE-NEXT:        %31 = "scf.if"(%30) ({
// SHAPE-NEXT:          %32 = stencil.access %arg10[0, -1, 0] : !stencil.temp<[0,64]x[-1,64]x[0,64]xf64>
// SHAPE-NEXT:          %33 = stencil.access %arg11[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %34 = arith.mulf %29, %33 : f64
// SHAPE-NEXT:          %35 = arith.subf %32, %34 : f64
// SHAPE-NEXT:          %36 = arith.subf %cst_1, %29 : f64
// SHAPE-NEXT:          %37 = arith.mulf %36, %35 : f64
// SHAPE-NEXT:          scf.yield %37 : f64
// SHAPE-NEXT:        }, {
// SHAPE-NEXT:          %38 = stencil.access %arg9[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:          %39 = stencil.access %arg11[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %40 = arith.mulf %29, %39 : f64
// SHAPE-NEXT:          %41 = arith.addf %38, %40 : f64
// SHAPE-NEXT:          %42 = arith.addf %cst_1, %29 : f64
// SHAPE-NEXT:          %43 = arith.mulf %42, %41 : f64
// SHAPE-NEXT:          scf.yield %43 : f64
// SHAPE-NEXT:        }) : (i1) -> f64
// SHAPE-NEXT:        %44 = arith.mulf %31, %28 : f64
// SHAPE-NEXT:        %45 = "scf.if"(%30) ({
// SHAPE-NEXT:          %46 = stencil.access %arg7[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %47 = arith.addf %46, %44 : f64
// SHAPE-NEXT:          scf.yield %47 : f64
// SHAPE-NEXT:        }, {
// SHAPE-NEXT:          %48 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %49 = arith.addf %48, %44 : f64
// SHAPE-NEXT:          scf.yield %49 : f64
// SHAPE-NEXT:        }) : (i1) -> f64
// SHAPE-NEXT:        %50 = stencil.store_result %45 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %50 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %23 = stencil.apply(%24 = %10 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %25 = %22 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,65]x[0,64]xf64>) {
// SHAPE-NEXT:        %26 = stencil.access %24[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %27 = stencil.access %25[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %28 = arith.mulf %26, %27 : f64
// SHAPE-NEXT:        %29 = stencil.store_result %28 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %29 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %24 = stencil.buffer %23 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:      %25 = stencil.apply(%arg7 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %arg8 = %11 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>, %arg9 = %24 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %arg10 = %9 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// SHAPE-NEXT:        %26 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:        %27 = stencil.access %arg8[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:        %28 = arith.mulf %26, %27 : f64
// SHAPE-NEXT:        %29 = stencil.access %arg9[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %30 = stencil.access %arg9[0, 1, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %31 = arith.subf %29, %30 : f64
// SHAPE-NEXT:        %32 = arith.addf %28, %31 : f64
// SHAPE-NEXT:        %33 = stencil.access %arg10[0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// SHAPE-NEXT:        %34 = arith.divf %32, %33 : f64
// SHAPE-NEXT:        %35 = stencil.store_result %34 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %35 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      stencil.store %22 to %6 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,65]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      stencil.store %25 to %5 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
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
// MLIR-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:        %90 = arith.constant -1 : index
// MLIR-NEXT:        %91 = arith.addi %88, %90 : index
// MLIR-NEXT:        %92 = memref.load %arg12[%87, %91, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        %93 = arith.cmpf oeq, %92, %cst : f64
// MLIR-NEXT:        %94 = arith.select %93, %cst_1, %cst : f64
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
// MLIR-NEXT:          %109 = arith.subf %cst_1, %98 : f64
// MLIR-NEXT:          %110 = arith.mulf %109, %108 : f64
// MLIR-NEXT:          scf.yield %110 : f64
// MLIR-NEXT:        }, {
// MLIR-NEXT:          %111 = memref.load %arg9[%87, %88, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %112 = memref.load %arg11[%87, %88, %89] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %113 = arith.mulf %98, %112 : f64
// MLIR-NEXT:          %114 = arith.addf %111, %113 : f64
// MLIR-NEXT:          %115 = arith.addf %cst_1, %98 : f64
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
