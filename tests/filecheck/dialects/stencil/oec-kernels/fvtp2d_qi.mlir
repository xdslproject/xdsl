// RUN: XDSL_ROUNDTRIP
// RUN: xdsl-opt %s -p stencil-storage-materialization,shape-inference | filecheck %s --check-prefix SHAPE
// RUN: xdsl-opt %s -p stencil-storage-materialization,shape-inference,convert-stencil-to-ll-mlir | filecheck %s --check-prefix MLIR
// RUN: xdsl-opt %s -p stencil-storage-materialization,shape-inference,stencil-bufferize | filecheck %s --check-prefix BUFF

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
    %25 = scf.if %24 -> (f64) {
      %29 = stencil.access %arg10 [0, -1, 0] : !stencil.temp<?x?x?xf64>
      %30 = stencil.access %arg11 [0, -1, 0] : !stencil.temp<?x?x?xf64>
      %31 = arith.mulf %23, %30 : f64
      %32 = arith.subf %29, %31 : f64
      %33 = arith.subf %cst_0, %23 : f64
      %34 = arith.mulf %33, %32 : f64
      scf.yield %34 : f64
    } else {
      %29 = stencil.access %arg9 [0, 0, 0] : !stencil.temp<?x?x?xf64>
      %30 = stencil.access %arg11 [0, 0, 0] : !stencil.temp<?x?x?xf64>
      %31 = arith.mulf %23, %30 : f64
      %32 = arith.addf %29, %31 : f64
      %33 = arith.addf %cst_0, %23 : f64
      %34 = arith.mulf %33, %32 : f64
      scf.yield %34 : f64
    }
    %26 = arith.mulf %25, %22 : f64
    %27 = scf.if %24 -> (f64) {
      %29 = stencil.access %arg7 [0, -1, 0] : !stencil.temp<?x?x?xf64>
      %30 = arith.addf %29, %26 : f64
      scf.yield %30 : f64
    } else {
      %29 = stencil.access %arg7 [0, 0, 0] : !stencil.temp<?x?x?xf64>
      %30 = arith.addf %29, %26 : f64
      scf.yield %30 : f64
    }
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
  stencil.store %14 to %6(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  stencil.store %16 to %5(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
  return
}

// CHECK:    func.func @fvtp2d_qi(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>, %arg3 : !stencil.field<?x?x?xf64>, %arg4 : !stencil.field<?x?x?xf64>, %arg5 : !stencil.field<?x?x?xf64>, %arg6 : !stencil.field<?x?x?xf64>) attributes {stencil.program} {
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
// CHECK-NEXT:        %26 = scf.if %25 -> (f64) {
// CHECK-NEXT:          %27 = stencil.access %arg10[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %28 = stencil.access %arg11[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %29 = arith.mulf %24, %28 : f64
// CHECK-NEXT:          %30 = arith.subf %27, %29 : f64
// CHECK-NEXT:          %31 = arith.subf %cst_1, %24 : f64
// CHECK-NEXT:          %32 = arith.mulf %31, %30 : f64
// CHECK-NEXT:          scf.yield %32 : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %33 = stencil.access %arg9[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %34 = stencil.access %arg11[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %35 = arith.mulf %24, %34 : f64
// CHECK-NEXT:          %36 = arith.addf %33, %35 : f64
// CHECK-NEXT:          %37 = arith.addf %cst_1, %24 : f64
// CHECK-NEXT:          %38 = arith.mulf %37, %36 : f64
// CHECK-NEXT:          scf.yield %38 : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        %39 = arith.mulf %26, %23 : f64
// CHECK-NEXT:        %40 = scf.if %25 -> (f64) {
// CHECK-NEXT:          %41 = stencil.access %arg7[0, -1, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %42 = arith.addf %41, %39 : f64
// CHECK-NEXT:          scf.yield %42 : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %43 = stencil.access %arg7[0, 0, 0] : !stencil.temp<?x?x?xf64>
// CHECK-NEXT:          %44 = arith.addf %43, %39 : f64
// CHECK-NEXT:          scf.yield %44 : f64
// CHECK-NEXT:        }
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
// CHECK-NEXT:      stencil.store %17 to %6(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      stencil.store %19 to %5(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<?x?x?xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// SHAPE:         func.func @fvtp2d_qi(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>, %arg3 : !stencil.field<?x?x?xf64>, %arg4 : !stencil.field<?x?x?xf64>, %arg5 : !stencil.field<?x?x?xf64>, %arg6 : !stencil.field<?x?x?xf64>) attributes {stencil.program} {
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
// SHAPE-NEXT:      %12 = stencil.apply(%arg7 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[-1,66]x[0,64]xf64>) {
// SHAPE-NEXT:        %cst = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:        %cst_1 = arith.constant 7.000000e+00 : f64
// SHAPE-NEXT:        %cst_2 = arith.constant 1.200000e+01 : f64
// SHAPE-NEXT:        %13 = arith.divf %cst_1, %cst_2 : f64
// SHAPE-NEXT:        %14 = arith.divf %cst, %cst_2 : f64
// SHAPE-NEXT:        %15 = stencil.access %arg7[0, -1, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %16 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %17 = arith.addf %15, %16 : f64
// SHAPE-NEXT:        %18 = stencil.access %arg7[0, -2, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %19 = stencil.access %arg7[0, 1, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %20 = arith.addf %18, %19 : f64
// SHAPE-NEXT:        %21 = arith.mulf %13, %17 : f64
// SHAPE-NEXT:        %22 = arith.mulf %14, %20 : f64
// SHAPE-NEXT:        %23 = arith.addf %21, %22 : f64
// SHAPE-NEXT:        %24 = stencil.store_result %23 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %24 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %13 = stencil.buffer %12 : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:      %14, %15, %16, %17 = stencil.apply(%arg7 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %arg8 = %13 : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>, !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>) {
// SHAPE-NEXT:        %cst = arith.constant 0.000000e+00 : f64
// SHAPE-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// SHAPE-NEXT:        %18 = stencil.access %arg8[0, 0, 0] : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:        %19 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:        %20 = arith.subf %18, %19 : f64
// SHAPE-NEXT:        %21 = stencil.access %arg8[0, 1, 0] : !stencil.temp<[0,64]x[-1,66]x[0,64]xf64>
// SHAPE-NEXT:        %22 = arith.subf %21, %19 : f64
// SHAPE-NEXT:        %23 = arith.addf %20, %22 : f64
// SHAPE-NEXT:        %24 = arith.mulf %20, %22 : f64
// SHAPE-NEXT:        %25 = arith.cmpf olt, %24, %cst : f64
// SHAPE-NEXT:        %26 = arith.select %25, %cst_1, %cst : f64
// SHAPE-NEXT:        %27 = stencil.store_result %20 : !stencil.result<f64>
// SHAPE-NEXT:        %28 = stencil.store_result %22 : !stencil.result<f64>
// SHAPE-NEXT:        %29 = stencil.store_result %23 : !stencil.result<f64>
// SHAPE-NEXT:        %30 = stencil.store_result %26 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %27, %28, %29, %30 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
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
// SHAPE-NEXT:        %31 = scf.if %30 -> (f64) {
// SHAPE-NEXT:          %32 = stencil.access %arg10[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %33 = stencil.access %arg11[0, -1, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %34 = arith.mulf %29, %33 : f64
// SHAPE-NEXT:          %35 = arith.subf %32, %34 : f64
// SHAPE-NEXT:          %36 = arith.subf %cst_1, %29 : f64
// SHAPE-NEXT:          %37 = arith.mulf %36, %35 : f64
// SHAPE-NEXT:          scf.yield %37 : f64
// SHAPE-NEXT:        } else {
// SHAPE-NEXT:          %38 = stencil.access %arg9[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %39 = stencil.access %arg11[0, 0, 0] : !stencil.temp<[0,64]x[-1,65]x[0,64]xf64>
// SHAPE-NEXT:          %40 = arith.mulf %29, %39 : f64
// SHAPE-NEXT:          %41 = arith.addf %38, %40 : f64
// SHAPE-NEXT:          %42 = arith.addf %cst_1, %29 : f64
// SHAPE-NEXT:          %43 = arith.mulf %42, %41 : f64
// SHAPE-NEXT:          scf.yield %43 : f64
// SHAPE-NEXT:        }
// SHAPE-NEXT:        %44 = arith.mulf %31, %28 : f64
// SHAPE-NEXT:        %45 = scf.if %30 -> (f64) {
// SHAPE-NEXT:          %46 = stencil.access %arg7[0, -1, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:          %47 = arith.addf %46, %44 : f64
// SHAPE-NEXT:          scf.yield %47 : f64
// SHAPE-NEXT:        } else {
// SHAPE-NEXT:          %48 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
// SHAPE-NEXT:          %49 = arith.addf %48, %44 : f64
// SHAPE-NEXT:          scf.yield %49 : f64
// SHAPE-NEXT:        }
// SHAPE-NEXT:        %50 = stencil.store_result %45 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %50 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %23 = stencil.apply(%arg7 = %10 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %arg8 = %22 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,65]x[0,64]xf64>) {
// SHAPE-NEXT:        %24 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %25 = stencil.access %arg8[0, 0, 0] : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:        %26 = arith.mulf %24, %25 : f64
// SHAPE-NEXT:        %27 = stencil.store_result %26 : !stencil.result<f64>
// SHAPE-NEXT:        stencil.return %27 : !stencil.result<f64>
// SHAPE-NEXT:      }
// SHAPE-NEXT:      %24 = stencil.buffer %23 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>
// SHAPE-NEXT:      %25 = stencil.apply(%arg7 = %7 : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>, %arg8 = %11 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>, %arg9 = %24 : !stencil.temp<[0,64]x[0,65]x[0,64]xf64>, %arg10 = %9 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// SHAPE-NEXT:        %26 = stencil.access %arg7[0, 0, 0] : !stencil.temp<[0,64]x[-3,67]x[0,64]xf64>
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
// SHAPE-NEXT:      stencil.store %22 to %6(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,65]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      stencil.store %25 to %5(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// SHAPE-NEXT:      func.return
// SHAPE-NEXT:    }

// MLIR:         func.func @fvtp2d_qi(%arg0 : memref<?x?x?xf64>, %arg1 : memref<?x?x?xf64>, %arg2 : memref<?x?x?xf64>, %arg3 : memref<?x?x?xf64>, %arg4 : memref<?x?x?xf64>, %arg5 : memref<?x?x?xf64>, %arg6 : memref<?x?x?xf64>) attributes {stencil.program} {
// MLIR-NEXT:      %arg8 = memref.alloc() : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:      %arg9 = memref.alloc() : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      %arg10 = memref.alloc() : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      %arg11 = memref.alloc() : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      %arg12 = memref.alloc() : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      %arg9_1 = memref.alloc() : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:      %0 = "memref.cast"(%arg0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %1 = "memref.cast"(%arg1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %2 = "memref.cast"(%arg2) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %3 = "memref.cast"(%arg3) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %4 = "memref.cast"(%arg4) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %5 = "memref.cast"(%arg5) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %6 = memref.subview %5[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %7 = "memref.cast"(%arg6) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// MLIR-NEXT:      %arg8_1 = memref.subview %7[4, 4, 4] [64, 65, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %8 = memref.subview %0[4, 4, 4] [64, 70, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %9 = memref.subview %1[4, 4, 4] [64, 65, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %10 = memref.subview %2[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %11 = memref.subview %3[4, 4, 4] [64, 65, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %12 = memref.subview %4[4, 4, 4] [64, 64, 64] [1, 1, 1] : memref<72x72x72xf64> to memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:      %13 = arith.constant 0 : index
// MLIR-NEXT:      %14 = arith.constant -1 : index
// MLIR-NEXT:      %15 = arith.constant 0 : index
// MLIR-NEXT:      %16 = arith.constant 1 : index
// MLIR-NEXT:      %17 = arith.constant 1 : index
// MLIR-NEXT:      %18 = arith.constant 1 : index
// MLIR-NEXT:      %19 = arith.constant 64 : index
// MLIR-NEXT:      %20 = arith.constant 66 : index
// MLIR-NEXT:      %21 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%13, %14, %15, %19, %20, %21, %16, %17, %18) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^0(%22 : index, %23 : index, %24 : index):
// MLIR-NEXT:        %cst = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:        %cst_1 = arith.constant 7.000000e+00 : f64
// MLIR-NEXT:        %cst_2 = arith.constant 1.200000e+01 : f64
// MLIR-NEXT:        %25 = arith.divf %cst_1, %cst_2 : f64
// MLIR-NEXT:        %26 = arith.divf %cst, %cst_2 : f64
// MLIR-NEXT:        %27 = arith.constant -1 : index
// MLIR-NEXT:        %28 = arith.addi %23, %27 : index
// MLIR-NEXT:        %29 = memref.load %8[%22, %28, %24] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %30 = memref.load %8[%22, %23, %24] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %31 = arith.addf %29, %30 : f64
// MLIR-NEXT:        %32 = arith.constant -2 : index
// MLIR-NEXT:        %33 = arith.addi %23, %32 : index
// MLIR-NEXT:        %34 = memref.load %8[%22, %33, %24] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %35 = arith.constant 1 : index
// MLIR-NEXT:        %36 = arith.addi %23, %35 : index
// MLIR-NEXT:        %37 = memref.load %8[%22, %36, %24] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %38 = arith.addf %34, %37 : f64
// MLIR-NEXT:        %39 = arith.mulf %25, %31 : f64
// MLIR-NEXT:        %40 = arith.mulf %26, %38 : f64
// MLIR-NEXT:        %41 = arith.addf %39, %40 : f64
// MLIR-NEXT:        memref.store %41, %arg8[%22, %23, %24] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:        scf.reduce
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      %42 = arith.constant 0 : index
// MLIR-NEXT:      %43 = arith.constant -1 : index
// MLIR-NEXT:      %44 = arith.constant 0 : index
// MLIR-NEXT:      %45 = arith.constant 1 : index
// MLIR-NEXT:      %46 = arith.constant 1 : index
// MLIR-NEXT:      %47 = arith.constant 1 : index
// MLIR-NEXT:      %48 = arith.constant 64 : index
// MLIR-NEXT:      %49 = arith.constant 65 : index
// MLIR-NEXT:      %50 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%42, %43, %44, %48, %49, %50, %45, %46, %47) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^1(%51 : index, %52 : index, %53 : index):
// MLIR-NEXT:        %cst_3 = arith.constant 0.000000e+00 : f64
// MLIR-NEXT:        %cst_4 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:        %54 = memref.load %arg8[%51, %52, %53] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:        %55 = memref.load %8[%51, %52, %53] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %56 = arith.subf %54, %55 : f64
// MLIR-NEXT:        %57 = arith.constant 1 : index
// MLIR-NEXT:        %58 = arith.addi %52, %57 : index
// MLIR-NEXT:        %59 = memref.load %arg8[%51, %58, %53] : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:        %60 = arith.subf %59, %55 : f64
// MLIR-NEXT:        %61 = arith.addf %56, %60 : f64
// MLIR-NEXT:        %62 = arith.mulf %56, %60 : f64
// MLIR-NEXT:        %63 = arith.cmpf olt, %62, %cst_3 : f64
// MLIR-NEXT:        %64 = arith.select %63, %cst_4, %cst_3 : f64
// MLIR-NEXT:        memref.store %56, %arg9[%51, %52, %53] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        memref.store %60, %arg10[%51, %52, %53] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        memref.store %61, %arg11[%51, %52, %53] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        memref.store %64, %arg12[%51, %52, %53] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        scf.reduce
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      %65 = arith.constant 0 : index
// MLIR-NEXT:      %66 = arith.constant 0 : index
// MLIR-NEXT:      %67 = arith.constant 0 : index
// MLIR-NEXT:      %68 = arith.constant 1 : index
// MLIR-NEXT:      %69 = arith.constant 1 : index
// MLIR-NEXT:      %70 = arith.constant 1 : index
// MLIR-NEXT:      %71 = arith.constant 64 : index
// MLIR-NEXT:      %72 = arith.constant 65 : index
// MLIR-NEXT:      %73 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%65, %66, %67, %71, %72, %73, %68, %69, %70) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^2(%74 : index, %75 : index, %76 : index):
// MLIR-NEXT:        %cst_5 = arith.constant 0.000000e+00 : f64
// MLIR-NEXT:        %cst_6 = arith.constant 1.000000e+00 : f64
// MLIR-NEXT:        %77 = arith.constant -1 : index
// MLIR-NEXT:        %78 = arith.addi %75, %77 : index
// MLIR-NEXT:        %79 = memref.load %arg12[%74, %78, %76] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        %80 = arith.cmpf oeq, %79, %cst_5 : f64
// MLIR-NEXT:        %81 = arith.select %80, %cst_6, %cst_5 : f64
// MLIR-NEXT:        %82 = memref.load %arg12[%74, %75, %76] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:        %83 = arith.mulf %82, %81 : f64
// MLIR-NEXT:        %84 = arith.addf %79, %83 : f64
// MLIR-NEXT:        %85 = memref.load %9[%74, %75, %76] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %86 = arith.cmpf ogt, %85, %cst_5 : f64
// MLIR-NEXT:        %87 = scf.if %86 -> (f64) {
// MLIR-NEXT:          %88 = arith.constant -1 : index
// MLIR-NEXT:          %89 = arith.addi %75, %88 : index
// MLIR-NEXT:          %90 = memref.load %arg10[%74, %89, %76] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %91 = arith.constant -1 : index
// MLIR-NEXT:          %92 = arith.addi %75, %91 : index
// MLIR-NEXT:          %93 = memref.load %arg11[%74, %92, %76] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %94 = arith.mulf %85, %93 : f64
// MLIR-NEXT:          %95 = arith.subf %90, %94 : f64
// MLIR-NEXT:          %96 = arith.subf %cst_6, %85 : f64
// MLIR-NEXT:          %97 = arith.mulf %96, %95 : f64
// MLIR-NEXT:          scf.yield %97 : f64
// MLIR-NEXT:        } else {
// MLIR-NEXT:          %98 = memref.load %arg9[%74, %75, %76] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %99 = memref.load %arg11[%74, %75, %76] : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:          %100 = arith.mulf %85, %99 : f64
// MLIR-NEXT:          %101 = arith.addf %98, %100 : f64
// MLIR-NEXT:          %102 = arith.addf %cst_6, %85 : f64
// MLIR-NEXT:          %103 = arith.mulf %102, %101 : f64
// MLIR-NEXT:          scf.yield %103 : f64
// MLIR-NEXT:        }
// MLIR-NEXT:        %104 = arith.mulf %87, %84 : f64
// MLIR-NEXT:        %105 = scf.if %86 -> (f64) {
// MLIR-NEXT:          %106 = arith.constant -1 : index
// MLIR-NEXT:          %107 = arith.addi %75, %106 : index
// MLIR-NEXT:          %108 = memref.load %8[%74, %107, %76] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:          %109 = arith.addf %108, %104 : f64
// MLIR-NEXT:          scf.yield %109 : f64
// MLIR-NEXT:        } else {
// MLIR-NEXT:          %110 = memref.load %8[%74, %75, %76] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:          %111 = arith.addf %110, %104 : f64
// MLIR-NEXT:          scf.yield %111 : f64
// MLIR-NEXT:        }
// MLIR-NEXT:        memref.store %105, %arg8_1[%74, %75, %76] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        scf.reduce
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      %112 = arith.constant 0 : index
// MLIR-NEXT:      %113 = arith.constant 0 : index
// MLIR-NEXT:      %114 = arith.constant 0 : index
// MLIR-NEXT:      %115 = arith.constant 1 : index
// MLIR-NEXT:      %116 = arith.constant 1 : index
// MLIR-NEXT:      %117 = arith.constant 1 : index
// MLIR-NEXT:      %118 = arith.constant 64 : index
// MLIR-NEXT:      %119 = arith.constant 65 : index
// MLIR-NEXT:      %120 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%112, %113, %114, %118, %119, %120, %115, %116, %117) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^3(%121 : index, %122 : index, %123 : index):
// MLIR-NEXT:        %124 = memref.load %11[%121, %122, %123] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %125 = memref.load %arg8_1[%121, %122, %123] : memref<64x65x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %126 = arith.mulf %124, %125 : f64
// MLIR-NEXT:        memref.store %126, %arg9_1[%121, %122, %123] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:        scf.reduce
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      %127 = arith.constant 0 : index
// MLIR-NEXT:      %128 = arith.constant 0 : index
// MLIR-NEXT:      %129 = arith.constant 0 : index
// MLIR-NEXT:      %130 = arith.constant 1 : index
// MLIR-NEXT:      %131 = arith.constant 1 : index
// MLIR-NEXT:      %132 = arith.constant 1 : index
// MLIR-NEXT:      %133 = arith.constant 64 : index
// MLIR-NEXT:      %134 = arith.constant 64 : index
// MLIR-NEXT:      %135 = arith.constant 64 : index
// MLIR-NEXT:      "scf.parallel"(%127, %128, %129, %133, %134, %135, %130, %131, %132) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// MLIR-NEXT:      ^4(%136 : index, %137 : index, %138 : index):
// MLIR-NEXT:        %139 = memref.load %8[%136, %137, %138] : memref<64x70x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %140 = memref.load %12[%136, %137, %138] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %141 = arith.mulf %139, %140 : f64
// MLIR-NEXT:        %142 = memref.load %arg9_1[%136, %137, %138] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:        %143 = arith.constant 1 : index
// MLIR-NEXT:        %144 = arith.addi %137, %143 : index
// MLIR-NEXT:        %145 = memref.load %arg9_1[%136, %144, %138] : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:        %146 = arith.subf %142, %145 : f64
// MLIR-NEXT:        %147 = arith.addf %141, %146 : f64
// MLIR-NEXT:        %148 = memref.load %10[%136, %137, %138] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        %149 = arith.divf %147, %148 : f64
// MLIR-NEXT:        memref.store %149, %6[%136, %137, %138] : memref<64x64x64xf64, strided<[5184, 72, 1], offset: 21028>>
// MLIR-NEXT:        scf.reduce
// MLIR-NEXT:      }) : (index, index, index, index, index, index, index, index, index) -> ()
// MLIR-NEXT:      memref.dealloc %arg9_1 : memref<64x65x64xf64, strided<[4160, 64, 1]>>
// MLIR-NEXT:      memref.dealloc %arg12 : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      memref.dealloc %arg11 : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      memref.dealloc %arg10 : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      memref.dealloc %arg9 : memref<64x66x64xf64, strided<[4224, 64, 1], offset: 64>>
// MLIR-NEXT:      memref.dealloc %arg8 : memref<64x67x64xf64, strided<[4288, 64, 1], offset: 64>>
// MLIR-NEXT:      func.return
// MLIR-NEXT:    }

// BUFF:         func.func @fvtp2d_qi(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>, %arg2 : !stencil.field<?x?x?xf64>, %arg3 : !stencil.field<?x?x?xf64>, %arg4 : !stencil.field<?x?x?xf64>, %arg5 : !stencil.field<?x?x?xf64>, %arg6 : !stencil.field<?x?x?xf64>) attributes {stencil.program} {
// BUFF-NEXT:      %0 = stencil.alloc : !stencil.field<[0,64]x[0,65]x[0,64]xf64>
// BUFF-NEXT:      %1 = stencil.alloc : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:      %2 = stencil.alloc : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:      %3 = stencil.alloc : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:      %4 = stencil.alloc : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:      %5 = stencil.alloc : !stencil.field<[0,64]x[-1,66]x[0,64]xf64>
// BUFF-NEXT:      %6 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:      %7 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:      %8 = stencil.cast %arg2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:      %9 = stencil.cast %arg3 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:      %10 = stencil.cast %arg4 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:      %11 = stencil.cast %arg5 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:      %12 = stencil.cast %arg6 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:      stencil.apply(%arg7 = %6 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) outs (%5 : !stencil.field<[0,64]x[-1,66]x[0,64]xf64>) {
// BUFF-NEXT:        %cst = arith.constant 1.000000e+00 : f64
// BUFF-NEXT:        %cst_1 = arith.constant 7.000000e+00 : f64
// BUFF-NEXT:        %cst_2 = arith.constant 1.200000e+01 : f64
// BUFF-NEXT:        %13 = arith.divf %cst_1, %cst_2 : f64
// BUFF-NEXT:        %14 = arith.divf %cst, %cst_2 : f64
// BUFF-NEXT:        %15 = stencil.access %arg7[0, -1, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %16 = stencil.access %arg7[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %17 = arith.addf %15, %16 : f64
// BUFF-NEXT:        %18 = stencil.access %arg7[0, -2, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %19 = stencil.access %arg7[0, 1, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %20 = arith.addf %18, %19 : f64
// BUFF-NEXT:        %21 = arith.mulf %13, %17 : f64
// BUFF-NEXT:        %22 = arith.mulf %14, %20 : f64
// BUFF-NEXT:        %23 = arith.addf %21, %22 : f64
// BUFF-NEXT:        %24 = stencil.store_result %23 : !stencil.result<f64>
// BUFF-NEXT:        stencil.return %24 : !stencil.result<f64>
// BUFF-NEXT:      } to <[0, -1, 0], [64, 66, 64]>
// BUFF-NEXT:      stencil.apply(%arg7 = %6 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %arg8 = %5 : !stencil.field<[0,64]x[-1,66]x[0,64]xf64>) outs (%4 : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>, %3 : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>, %2 : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>, %1 : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>) {
// BUFF-NEXT:        %cst = arith.constant 0.000000e+00 : f64
// BUFF-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// BUFF-NEXT:        %13 = stencil.access %arg8[0, 0, 0] : !stencil.field<[0,64]x[-1,66]x[0,64]xf64>
// BUFF-NEXT:        %14 = stencil.access %arg7[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %15 = arith.subf %13, %14 : f64
// BUFF-NEXT:        %16 = stencil.access %arg8[0, 1, 0] : !stencil.field<[0,64]x[-1,66]x[0,64]xf64>
// BUFF-NEXT:        %17 = arith.subf %16, %14 : f64
// BUFF-NEXT:        %18 = arith.addf %15, %17 : f64
// BUFF-NEXT:        %19 = arith.mulf %15, %17 : f64
// BUFF-NEXT:        %20 = arith.cmpf olt, %19, %cst : f64
// BUFF-NEXT:        %21 = arith.select %20, %cst_1, %cst : f64
// BUFF-NEXT:        %22 = stencil.store_result %15 : !stencil.result<f64>
// BUFF-NEXT:        %23 = stencil.store_result %17 : !stencil.result<f64>
// BUFF-NEXT:        %24 = stencil.store_result %18 : !stencil.result<f64>
// BUFF-NEXT:        %25 = stencil.store_result %21 : !stencil.result<f64>
// BUFF-NEXT:        stencil.return %22, %23, %24, %25 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// BUFF-NEXT:      } to <[0, -1, 0], [64, 65, 64]>
// BUFF-NEXT:      stencil.apply(%arg7 = %6 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %arg8 = %7 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %arg9 = %4 : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>, %arg10 = %3 : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>, %arg11 = %2 : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>, %arg12 = %1 : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>) outs (%12 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// BUFF-NEXT:        %cst = arith.constant 0.000000e+00 : f64
// BUFF-NEXT:        %cst_1 = arith.constant 1.000000e+00 : f64
// BUFF-NEXT:        %13 = stencil.access %arg12[0, -1, 0] : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:        %14 = arith.cmpf oeq, %13, %cst : f64
// BUFF-NEXT:        %15 = arith.select %14, %cst_1, %cst : f64
// BUFF-NEXT:        %16 = stencil.access %arg12[0, 0, 0] : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:        %17 = arith.mulf %16, %15 : f64
// BUFF-NEXT:        %18 = arith.addf %13, %17 : f64
// BUFF-NEXT:        %19 = stencil.access %arg8[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %20 = arith.cmpf ogt, %19, %cst : f64
// BUFF-NEXT:        %21 = scf.if %20 -> (f64) {
// BUFF-NEXT:          %22 = stencil.access %arg10[0, -1, 0] : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:          %23 = stencil.access %arg11[0, -1, 0] : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:          %24 = arith.mulf %19, %23 : f64
// BUFF-NEXT:          %25 = arith.subf %22, %24 : f64
// BUFF-NEXT:          %26 = arith.subf %cst_1, %19 : f64
// BUFF-NEXT:          %27 = arith.mulf %26, %25 : f64
// BUFF-NEXT:          scf.yield %27 : f64
// BUFF-NEXT:        } else {
// BUFF-NEXT:          %28 = stencil.access %arg9[0, 0, 0] : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:          %29 = stencil.access %arg11[0, 0, 0] : !stencil.field<[0,64]x[-1,65]x[0,64]xf64>
// BUFF-NEXT:          %30 = arith.mulf %19, %29 : f64
// BUFF-NEXT:          %31 = arith.addf %28, %30 : f64
// BUFF-NEXT:          %32 = arith.addf %cst_1, %19 : f64
// BUFF-NEXT:          %33 = arith.mulf %32, %31 : f64
// BUFF-NEXT:          scf.yield %33 : f64
// BUFF-NEXT:        }
// BUFF-NEXT:        %34 = arith.mulf %21, %18 : f64
// BUFF-NEXT:        %35 = scf.if %20 -> (f64) {
// BUFF-NEXT:          %36 = stencil.access %arg7[0, -1, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:          %37 = arith.addf %36, %34 : f64
// BUFF-NEXT:          scf.yield %37 : f64
// BUFF-NEXT:        } else {
// BUFF-NEXT:          %38 = stencil.access %arg7[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:          %39 = arith.addf %38, %34 : f64
// BUFF-NEXT:          scf.yield %39 : f64
// BUFF-NEXT:        }
// BUFF-NEXT:        %40 = stencil.store_result %35 : !stencil.result<f64>
// BUFF-NEXT:        stencil.return %40 : !stencil.result<f64>
// BUFF-NEXT:      } to <[0, 0, 0], [64, 65, 64]>
// BUFF-NEXT:      stencil.apply(%arg7 = %9 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %arg8 = %12 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) outs (%0 : !stencil.field<[0,64]x[0,65]x[0,64]xf64>) {
// BUFF-NEXT:        %13 = stencil.access %arg7[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %14 = stencil.access %arg8[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %15 = arith.mulf %13, %14 : f64
// BUFF-NEXT:        %16 = stencil.store_result %15 : !stencil.result<f64>
// BUFF-NEXT:        stencil.return %16 : !stencil.result<f64>
// BUFF-NEXT:      } to <[0, 0, 0], [64, 65, 64]>
// BUFF-NEXT:      stencil.apply(%arg7 = %6 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %arg8 = %10 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>, %arg9 = %0 : !stencil.field<[0,64]x[0,65]x[0,64]xf64>, %arg10 = %8 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) outs (%11 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) {
// BUFF-NEXT:        %13 = stencil.access %arg7[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %14 = stencil.access %arg8[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %15 = arith.mulf %13, %14 : f64
// BUFF-NEXT:        %16 = stencil.access %arg9[0, 0, 0] : !stencil.field<[0,64]x[0,65]x[0,64]xf64>
// BUFF-NEXT:        %17 = stencil.access %arg9[0, 1, 0] : !stencil.field<[0,64]x[0,65]x[0,64]xf64>
// BUFF-NEXT:        %18 = arith.subf %16, %17 : f64
// BUFF-NEXT:        %19 = arith.addf %15, %18 : f64
// BUFF-NEXT:        %20 = stencil.access %arg10[0, 0, 0] : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// BUFF-NEXT:        %21 = arith.divf %19, %20 : f64
// BUFF-NEXT:        %22 = stencil.store_result %21 : !stencil.result<f64>
// BUFF-NEXT:        stencil.return %22 : !stencil.result<f64>
// BUFF-NEXT:      } to <[0, 0, 0], [64, 64, 64]>
// BUFF-NEXT:      func.return
// BUFF-NEXT:    }
