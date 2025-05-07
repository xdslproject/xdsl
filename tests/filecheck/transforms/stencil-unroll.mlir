// RUN: xdsl-opt %s -p "stencil-unroll{unroll-factor=8,1}" | filecheck %s

  func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
      %5 = arith.constant 1.000000e+00 : f64
      %6 = arith.addf %4, %5 : f64
      stencil.return %6 : f64
    }
    stencil.store %3 to %2(<[1, 2, 3], [65, 66, 63]>) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK:         func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
// CHECK-NEXT:        %5 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %6 = arith.addf %4, %5 : f64
// CHECK-NEXT:        %7 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %8 = arith.addf %4, %7 : f64
// CHECK-NEXT:        %9 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %10 = arith.addf %4, %9 : f64
// CHECK-NEXT:        %11 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %12 = arith.addf %4, %11 : f64
// CHECK-NEXT:        %13 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %14 = arith.addf %4, %13 : f64
// CHECK-NEXT:        %15 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %16 = arith.addf %4, %15 : f64
// CHECK-NEXT:        %17 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %18 = arith.addf %4, %17 : f64
// CHECK-NEXT:        %19 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %20 = arith.addf %4, %19 : f64
// CHECK-NEXT:        stencil.return %6, %8, %10, %12, %14, %16, %18, %20 unroll <[1, 8, 1]> : f64, f64, f64, f64, f64, f64, f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %3 to %2(<[1, 2, 3], [65, 66, 63]>) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @copy_1d(%7 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
    %8 = stencil.cast %7 : !stencil.field<?xf64> -> !stencil.field<[-4,68]xf64>
    %outc = stencil.cast %out : !stencil.field<?xf64> -> !stencil.field<[0,1024]xf64>
    %9 = stencil.load %8 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,68]xf64>
    %10 = stencil.apply(%11 = %9 : !stencil.temp<[-1,68]xf64>) -> (!stencil.temp<[0,68]xf64>) {
      %12 = stencil.access %11[-1] : !stencil.temp<[-1,68]xf64>
      stencil.return %12 : f64
    }
    stencil.store %10 to %outc(<[0], [68]>) : !stencil.temp<[0,68]xf64> to !stencil.field<[0,1024]xf64>
    func.return
  }

// CHECK:         func.func @copy_1d(%0 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
// CHECK-NEXT:      %1 = stencil.cast %0 : !stencil.field<?xf64> -> !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      %outc = stencil.cast %out : !stencil.field<?xf64> -> !stencil.field<[0,1024]xf64>
// CHECK-NEXT:      %2 = stencil.load %1 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,68]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%4 = %2 : !stencil.temp<[-1,68]xf64>) -> (!stencil.temp<[0,68]xf64>) {
// CHECK-NEXT:        %5 = stencil.access %4[-1] : !stencil.temp<[-1,68]xf64>
// CHECK-NEXT:        stencil.return %5 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %3 to %outc(<[0], [68]>) : !stencil.temp<[0,68]xf64> to !stencil.field<[0,1024]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @offsets(%13 : !stencil.field<?x?x?xf64>, %14 : !stencil.field<?x?x?xf64>, %15 : !stencil.field<?x?x?xf64>) {
    %16 = stencil.cast %13 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %17 = stencil.cast %14 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %18 = stencil.cast %15 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %19 = stencil.load %16 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %20, %21 = stencil.apply(%22 = %19 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
      %23 = stencil.access %22[-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %24 = stencil.access %22[1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %25 = stencil.access %22[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %26 = stencil.access %22[0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %27 = stencil.access %22[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
      %28 = arith.addf %23, %24 : f64
      %29 = arith.addf %25, %26 : f64
      %30 = arith.addf %28, %29 : f64
      %cst = arith.constant -4.000000e+00 : f64
      %31 = arith.mulf %27, %cst : f64
      %32 = arith.addf %31, %30 : f64
      stencil.return %32, %31 : f64, f64
    }
    stencil.store %20 to %17(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @offsets(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>, %2 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %3 = stencil.cast %0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %4 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %5 = stencil.cast %2 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %6 = stencil.load %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:      %7, %8 = stencil.apply(%9 = %6 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %10 = stencil.access %9[-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %11 = stencil.access %9[1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %12 = stencil.access %9[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %13 = stencil.access %9[0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %14 = stencil.access %9[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %15 = arith.addf %10, %11 : f64
// CHECK-NEXT:        %16 = arith.addf %12, %13 : f64
// CHECK-NEXT:        %17 = arith.addf %15, %16 : f64
// CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %18 = arith.mulf %14, %cst : f64
// CHECK-NEXT:        %19 = arith.addf %18, %17 : f64
// CHECK-NEXT:        %20 = stencil.access %9[-1, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %21 = stencil.access %9[1, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %22 = stencil.access %9[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %23 = stencil.access %9[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %24 = stencil.access %9[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %25 = arith.addf %20, %21 : f64
// CHECK-NEXT:        %26 = arith.addf %22, %23 : f64
// CHECK-NEXT:        %27 = arith.addf %25, %26 : f64
// CHECK-NEXT:        %cst_1 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %28 = arith.mulf %24, %cst_1 : f64
// CHECK-NEXT:        %29 = arith.addf %28, %27 : f64
// CHECK-NEXT:        %30 = stencil.access %9[-1, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %31 = stencil.access %9[1, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %32 = stencil.access %9[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %33 = stencil.access %9[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %34 = stencil.access %9[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %35 = arith.addf %30, %31 : f64
// CHECK-NEXT:        %36 = arith.addf %32, %33 : f64
// CHECK-NEXT:        %37 = arith.addf %35, %36 : f64
// CHECK-NEXT:        %cst_2 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %38 = arith.mulf %34, %cst_2 : f64
// CHECK-NEXT:        %39 = arith.addf %38, %37 : f64
// CHECK-NEXT:        %40 = stencil.access %9[-1, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %41 = stencil.access %9[1, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %42 = stencil.access %9[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %43 = stencil.access %9[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %44 = stencil.access %9[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %45 = arith.addf %40, %41 : f64
// CHECK-NEXT:        %46 = arith.addf %42, %43 : f64
// CHECK-NEXT:        %47 = arith.addf %45, %46 : f64
// CHECK-NEXT:        %cst_3 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %48 = arith.mulf %44, %cst_3 : f64
// CHECK-NEXT:        %49 = arith.addf %48, %47 : f64
// CHECK-NEXT:        %50 = stencil.access %9[-1, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %51 = stencil.access %9[1, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %52 = stencil.access %9[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %53 = stencil.access %9[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %54 = stencil.access %9[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %55 = arith.addf %50, %51 : f64
// CHECK-NEXT:        %56 = arith.addf %52, %53 : f64
// CHECK-NEXT:        %57 = arith.addf %55, %56 : f64
// CHECK-NEXT:        %cst_4 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %58 = arith.mulf %54, %cst_4 : f64
// CHECK-NEXT:        %59 = arith.addf %58, %57 : f64
// CHECK-NEXT:        %60 = stencil.access %9[-1, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %61 = stencil.access %9[1, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %62 = stencil.access %9[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %63 = stencil.access %9[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %64 = stencil.access %9[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %65 = arith.addf %60, %61 : f64
// CHECK-NEXT:        %66 = arith.addf %62, %63 : f64
// CHECK-NEXT:        %67 = arith.addf %65, %66 : f64
// CHECK-NEXT:        %cst_5 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %68 = arith.mulf %64, %cst_5 : f64
// CHECK-NEXT:        %69 = arith.addf %68, %67 : f64
// CHECK-NEXT:        %70 = stencil.access %9[-1, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %71 = stencil.access %9[1, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %72 = stencil.access %9[0, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %73 = stencil.access %9[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %74 = stencil.access %9[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %75 = arith.addf %70, %71 : f64
// CHECK-NEXT:        %76 = arith.addf %72, %73 : f64
// CHECK-NEXT:        %77 = arith.addf %75, %76 : f64
// CHECK-NEXT:        %cst_6 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %78 = arith.mulf %74, %cst_6 : f64
// CHECK-NEXT:        %79 = arith.addf %78, %77 : f64
// CHECK-NEXT:        %80 = stencil.access %9[-1, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %81 = stencil.access %9[1, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %82 = stencil.access %9[0, 8, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %83 = stencil.access %9[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %84 = stencil.access %9[0, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %85 = arith.addf %80, %81 : f64
// CHECK-NEXT:        %86 = arith.addf %82, %83 : f64
// CHECK-NEXT:        %87 = arith.addf %85, %86 : f64
// CHECK-NEXT:        %cst_7 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %88 = arith.mulf %84, %cst_7 : f64
// CHECK-NEXT:        %89 = arith.addf %88, %87 : f64
// CHECK-NEXT:        stencil.return %19, %18, %29, %28, %39, %38, %49, %48, %59, %58, %69, %68, %79, %78, %89, %88 unroll <[1, 8, 1]> : f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %7 to %4(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes {"stencil.program"} {
    %33 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %34 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %35 = stencil.load %33 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %36 = stencil.apply(%arg2 = %35 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>)  attributes {"lb" = #stencil.index<[0, 0, 0]>, "ub" = [64 : i64, 64 : i64, 60 : i64]} {
      %37 = stencil.index 0 <[0, 0, 0]>
      %38 = stencil.index 1 <[0, 0, 0]>
      %39 = stencil.index 2 <[0, 0, 0]>
      %40 = stencil.dyn_access %arg2[%37, %38, %39] in <[0, 0, 0]> : <[0, 0, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
      %41 = stencil.store_result %40 : !stencil.result<f64>
      stencil.return %41 : !stencil.result<f64>
    }
    stencil.store %36 to %34(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    func.return
  }

// CHECK-NEXT:    func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes {stencil.program} {
// CHECK-NEXT:      %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %2 = stencil.load %0 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%arg2 = %2 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %4 = stencil.index 0 <[0, 0, 0]>
// CHECK-NEXT:        %5 = stencil.index 1 <[0, 0, 0]>
// CHECK-NEXT:        %6 = stencil.index 2 <[0, 0, 0]>
// CHECK-NEXT:        %7 = stencil.dyn_access %arg2[%4, %5, %6] in <[0, 0, 0]> : <[0, 0, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %8 = stencil.store_result %7 : !stencil.result<f64>
// CHECK-NEXT:        %9 = stencil.index 0 <[0, 1, 0]>
// CHECK-NEXT:        %10 = stencil.index 1 <[0, 1, 0]>
// CHECK-NEXT:        %11 = stencil.index 2 <[0, 1, 0]>
// CHECK-NEXT:        %12 = stencil.dyn_access %arg2[%9, %10, %11] in <[0, 1, 0]> : <[0, 1, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %13 = stencil.store_result %12 : !stencil.result<f64>
// CHECK-NEXT:        %14 = stencil.index 0 <[0, 2, 0]>
// CHECK-NEXT:        %15 = stencil.index 1 <[0, 2, 0]>
// CHECK-NEXT:        %16 = stencil.index 2 <[0, 2, 0]>
// CHECK-NEXT:        %17 = stencil.dyn_access %arg2[%14, %15, %16] in <[0, 2, 0]> : <[0, 2, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %18 = stencil.store_result %17 : !stencil.result<f64>
// CHECK-NEXT:        %19 = stencil.index 0 <[0, 3, 0]>
// CHECK-NEXT:        %20 = stencil.index 1 <[0, 3, 0]>
// CHECK-NEXT:        %21 = stencil.index 2 <[0, 3, 0]>
// CHECK-NEXT:        %22 = stencil.dyn_access %arg2[%19, %20, %21] in <[0, 3, 0]> : <[0, 3, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %23 = stencil.store_result %22 : !stencil.result<f64>
// CHECK-NEXT:        %24 = stencil.index 0 <[0, 4, 0]>
// CHECK-NEXT:        %25 = stencil.index 1 <[0, 4, 0]>
// CHECK-NEXT:        %26 = stencil.index 2 <[0, 4, 0]>
// CHECK-NEXT:        %27 = stencil.dyn_access %arg2[%24, %25, %26] in <[0, 4, 0]> : <[0, 4, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %28 = stencil.store_result %27 : !stencil.result<f64>
// CHECK-NEXT:        %29 = stencil.index 0 <[0, 5, 0]>
// CHECK-NEXT:        %30 = stencil.index 1 <[0, 5, 0]>
// CHECK-NEXT:        %31 = stencil.index 2 <[0, 5, 0]>
// CHECK-NEXT:        %32 = stencil.dyn_access %arg2[%29, %30, %31] in <[0, 5, 0]> : <[0, 5, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %33 = stencil.store_result %32 : !stencil.result<f64>
// CHECK-NEXT:        %34 = stencil.index 0 <[0, 6, 0]>
// CHECK-NEXT:        %35 = stencil.index 1 <[0, 6, 0]>
// CHECK-NEXT:        %36 = stencil.index 2 <[0, 6, 0]>
// CHECK-NEXT:        %37 = stencil.dyn_access %arg2[%34, %35, %36] in <[0, 6, 0]> : <[0, 6, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %38 = stencil.store_result %37 : !stencil.result<f64>
// CHECK-NEXT:        %39 = stencil.index 0 <[0, 7, 0]>
// CHECK-NEXT:        %40 = stencil.index 1 <[0, 7, 0]>
// CHECK-NEXT:        %41 = stencil.index 2 <[0, 7, 0]>
// CHECK-NEXT:        %42 = stencil.dyn_access %arg2[%39, %40, %41] in <[0, 7, 0]> : <[0, 7, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %43 = stencil.store_result %42 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %8, %13, %18, %23, %28, %33, %38, %43 unroll <[1, 8, 1]> : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %3 to %1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
