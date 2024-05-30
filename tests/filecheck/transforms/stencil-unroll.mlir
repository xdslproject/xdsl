// RUN: xdsl-opt %s -p "stencil-unroll{unroll-factor=8,1}" | filecheck %s

  func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
      %5 = arith.constant 1.000000e+00 : f64
      %6 = arith.addf %4, %5 : f64
      stencil.return %6 : f64
    }
    stencil.store %3 to %2 ([1, 2, 3] : [65, 66, 63]) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
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
// CHECK-NEXT:        stencil.return %6, %8, %10, %12, %14, %16, %18, %20 unroll [1, 8, 1] : f64, f64, f64, f64, f64, f64, f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %3 to %2 ([1, 2, 3] : [65, 66, 63]) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
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
    stencil.store %10 to %outc ([0] : [68]) : !stencil.temp<[0,68]xf64> to !stencil.field<[0,1024]xf64>
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
// CHECK-NEXT:      stencil.store %3 to %outc ([0] : [68]) : !stencil.temp<[0,68]xf64> to !stencil.field<[0,1024]xf64>
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
    stencil.store %20 to %17 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
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
// CHECK-NEXT:        %18 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %19 = arith.mulf %14, %18 : f64
// CHECK-NEXT:        %20 = arith.addf %19, %17 : f64
// CHECK-NEXT:        %21 = stencil.access %9[-1, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %22 = stencil.access %9[1, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %23 = stencil.access %9[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %24 = stencil.access %9[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %25 = stencil.access %9[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %26 = arith.addf %21, %22 : f64
// CHECK-NEXT:        %27 = arith.addf %23, %24 : f64
// CHECK-NEXT:        %28 = arith.addf %26, %27 : f64
// CHECK-NEXT:        %29 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %30 = arith.mulf %25, %29 : f64
// CHECK-NEXT:        %31 = arith.addf %30, %28 : f64
// CHECK-NEXT:        %32 = stencil.access %9[-1, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %33 = stencil.access %9[1, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %34 = stencil.access %9[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %35 = stencil.access %9[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %36 = stencil.access %9[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %37 = arith.addf %32, %33 : f64
// CHECK-NEXT:        %38 = arith.addf %34, %35 : f64
// CHECK-NEXT:        %39 = arith.addf %37, %38 : f64
// CHECK-NEXT:        %40 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %41 = arith.mulf %36, %40 : f64
// CHECK-NEXT:        %42 = arith.addf %41, %39 : f64
// CHECK-NEXT:        %43 = stencil.access %9[-1, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %44 = stencil.access %9[1, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %45 = stencil.access %9[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %46 = stencil.access %9[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %47 = stencil.access %9[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %48 = arith.addf %43, %44 : f64
// CHECK-NEXT:        %49 = arith.addf %45, %46 : f64
// CHECK-NEXT:        %50 = arith.addf %48, %49 : f64
// CHECK-NEXT:        %51 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %52 = arith.mulf %47, %51 : f64
// CHECK-NEXT:        %53 = arith.addf %52, %50 : f64
// CHECK-NEXT:        %54 = stencil.access %9[-1, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %55 = stencil.access %9[1, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %56 = stencil.access %9[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %57 = stencil.access %9[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %58 = stencil.access %9[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %59 = arith.addf %54, %55 : f64
// CHECK-NEXT:        %60 = arith.addf %56, %57 : f64
// CHECK-NEXT:        %61 = arith.addf %59, %60 : f64
// CHECK-NEXT:        %62 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %63 = arith.mulf %58, %62 : f64
// CHECK-NEXT:        %64 = arith.addf %63, %61 : f64
// CHECK-NEXT:        %65 = stencil.access %9[-1, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %66 = stencil.access %9[1, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %67 = stencil.access %9[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %68 = stencil.access %9[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %69 = stencil.access %9[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %70 = arith.addf %65, %66 : f64
// CHECK-NEXT:        %71 = arith.addf %67, %68 : f64
// CHECK-NEXT:        %72 = arith.addf %70, %71 : f64
// CHECK-NEXT:        %73 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %74 = arith.mulf %69, %73 : f64
// CHECK-NEXT:        %75 = arith.addf %74, %72 : f64
// CHECK-NEXT:        %76 = stencil.access %9[-1, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %77 = stencil.access %9[1, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %78 = stencil.access %9[0, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %79 = stencil.access %9[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %80 = stencil.access %9[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %81 = arith.addf %76, %77 : f64
// CHECK-NEXT:        %82 = arith.addf %78, %79 : f64
// CHECK-NEXT:        %83 = arith.addf %81, %82 : f64
// CHECK-NEXT:        %84 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %85 = arith.mulf %80, %84 : f64
// CHECK-NEXT:        %86 = arith.addf %85, %83 : f64
// CHECK-NEXT:        %87 = stencil.access %9[-1, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %88 = stencil.access %9[1, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %89 = stencil.access %9[0, 8, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %90 = stencil.access %9[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %91 = stencil.access %9[0, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %92 = arith.addf %87, %88 : f64
// CHECK-NEXT:        %93 = arith.addf %89, %90 : f64
// CHECK-NEXT:        %94 = arith.addf %92, %93 : f64
// CHECK-NEXT:        %95 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %96 = arith.mulf %91, %95 : f64
// CHECK-NEXT:        %97 = arith.addf %96, %94 : f64
// CHECK-NEXT:        stencil.return %20, %19, %31, %30, %42, %41, %53, %52, %64, %63, %75, %74, %86, %85, %97, %96 unroll [1, 8, 1] : f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %7 to %4 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
    %33 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %34 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    %35 = stencil.load %33 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %36 = stencil.apply(%arg2 = %35 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>)  attributes {"lb" = #stencil.index[0, 0, 0], "ub" = [64 : i64, 64 : i64, 60 : i64]}{
      %37 = stencil.index 0 [0, 0, 0]
      %38 = stencil.index 1 [0, 0, 0]
      %39 = stencil.index 2 [0, 0, 0]
      %40 = stencil.dyn_access %arg2[%37, %38, %39] in [0, 0, 0] : [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
      %41 = stencil.store_result %40 : !stencil.result<f64>
      stencil.return %41 : !stencil.result<f64>
    }
    stencil.store %36 to %34 ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
    func.return
  }

// CHECK:         func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:      %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %2 = stencil.load %0 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %3 = stencil.apply(%4 = %2 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %5 = stencil.index 0 [0, 0, 0]
// CHECK-NEXT:        %6 = stencil.index 1 [0, 0, 0]
// CHECK-NEXT:        %7 = stencil.index 2 [0, 0, 0]
// CHECK-NEXT:        %8 = stencil.dyn_access %4[%5, %6, %7] in [0, 0, 0] : [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %9 = stencil.store_result %8 : !stencil.result<f64>
// CHECK-NEXT:        %10 = stencil.index 0 [0, 1, 0]
// CHECK-NEXT:        %11 = stencil.index 1 [0, 1, 0]
// CHECK-NEXT:        %12 = stencil.index 2 [0, 1, 0]
// CHECK-NEXT:        %13 = stencil.dyn_access %4[%10, %11, %12] in [0, 1, 0] : [0, 1, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %14 = stencil.store_result %13 : !stencil.result<f64>
// CHECK-NEXT:        %15 = stencil.index 0 [0, 2, 0]
// CHECK-NEXT:        %16 = stencil.index 1 [0, 2, 0]
// CHECK-NEXT:        %17 = stencil.index 2 [0, 2, 0]
// CHECK-NEXT:        %18 = stencil.dyn_access %4[%15, %16, %17] in [0, 2, 0] : [0, 2, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %19 = stencil.store_result %18 : !stencil.result<f64>
// CHECK-NEXT:        %20 = stencil.index 0 [0, 3, 0]
// CHECK-NEXT:        %21 = stencil.index 1 [0, 3, 0]
// CHECK-NEXT:        %22 = stencil.index 2 [0, 3, 0]
// CHECK-NEXT:        %23 = stencil.dyn_access %4[%20, %21, %22] in [0, 3, 0] : [0, 3, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %24 = stencil.store_result %23 : !stencil.result<f64>
// CHECK-NEXT:        %25 = stencil.index 0 [0, 4, 0]
// CHECK-NEXT:        %26 = stencil.index 1 [0, 4, 0]
// CHECK-NEXT:        %27 = stencil.index 2 [0, 4, 0]
// CHECK-NEXT:        %28 = stencil.dyn_access %4[%25, %26, %27] in [0, 4, 0] : [0, 4, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %29 = stencil.store_result %28 : !stencil.result<f64>
// CHECK-NEXT:        %30 = stencil.index 0 [0, 5, 0]
// CHECK-NEXT:        %31 = stencil.index 1 [0, 5, 0]
// CHECK-NEXT:        %32 = stencil.index 2 [0, 5, 0]
// CHECK-NEXT:        %33 = stencil.dyn_access %4[%30, %31, %32] in [0, 5, 0] : [0, 5, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %34 = stencil.store_result %33 : !stencil.result<f64>
// CHECK-NEXT:        %35 = stencil.index 0 [0, 6, 0]
// CHECK-NEXT:        %36 = stencil.index 1 [0, 6, 0]
// CHECK-NEXT:        %37 = stencil.index 2 [0, 6, 0]
// CHECK-NEXT:        %38 = stencil.dyn_access %4[%35, %36, %37] in [0, 6, 0] : [0, 6, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %39 = stencil.store_result %38 : !stencil.result<f64>
// CHECK-NEXT:        %40 = stencil.index 0 [0, 7, 0]
// CHECK-NEXT:        %41 = stencil.index 1 [0, 7, 0]
// CHECK-NEXT:        %42 = stencil.index 2 [0, 7, 0]
// CHECK-NEXT:        %43 = stencil.dyn_access %4[%40, %41, %42] in [0, 7, 0] : [0, 7, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %44 = stencil.store_result %43 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %9, %14, %19, %24, %29, %34, %39, %44 unroll [1, 8, 1] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %3 to %1 ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
