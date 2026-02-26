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
// CHECK-NEXT:      %5 = stencil.load %3 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:      %6, %7 = stencil.apply(%8 = %5 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %9 = stencil.access %8[-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %10 = stencil.access %8[1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %11 = stencil.access %8[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %12 = stencil.access %8[0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %13 = stencil.access %8[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %14 = arith.addf %9, %10 : f64
// CHECK-NEXT:        %15 = arith.addf %11, %12 : f64
// CHECK-NEXT:        %16 = arith.addf %14, %15 : f64
// CHECK-NEXT:        %cst = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %17 = arith.mulf %13, %cst : f64
// CHECK-NEXT:        %18 = arith.addf %17, %16 : f64
// CHECK-NEXT:        %19 = stencil.access %8[-1, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %20 = stencil.access %8[1, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %21 = stencil.access %8[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %22 = stencil.access %8[0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %23 = stencil.access %8[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %24 = arith.addf %19, %20 : f64
// CHECK-NEXT:        %25 = arith.addf %21, %22 : f64
// CHECK-NEXT:        %26 = arith.addf %24, %25 : f64
// CHECK-NEXT:        %cst_1 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %27 = arith.mulf %23, %cst_1 : f64
// CHECK-NEXT:        %28 = arith.addf %27, %26 : f64
// CHECK-NEXT:        %29 = stencil.access %8[-1, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %30 = stencil.access %8[1, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %31 = stencil.access %8[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %32 = stencil.access %8[0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %33 = stencil.access %8[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %34 = arith.addf %29, %30 : f64
// CHECK-NEXT:        %35 = arith.addf %31, %32 : f64
// CHECK-NEXT:        %36 = arith.addf %34, %35 : f64
// CHECK-NEXT:        %cst_2 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %37 = arith.mulf %33, %cst_2 : f64
// CHECK-NEXT:        %38 = arith.addf %37, %36 : f64
// CHECK-NEXT:        %39 = stencil.access %8[-1, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %40 = stencil.access %8[1, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %41 = stencil.access %8[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %42 = stencil.access %8[0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %43 = stencil.access %8[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %44 = arith.addf %39, %40 : f64
// CHECK-NEXT:        %45 = arith.addf %41, %42 : f64
// CHECK-NEXT:        %46 = arith.addf %44, %45 : f64
// CHECK-NEXT:        %cst_3 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %47 = arith.mulf %43, %cst_3 : f64
// CHECK-NEXT:        %48 = arith.addf %47, %46 : f64
// CHECK-NEXT:        %49 = stencil.access %8[-1, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %50 = stencil.access %8[1, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %51 = stencil.access %8[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %52 = stencil.access %8[0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %53 = stencil.access %8[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %54 = arith.addf %49, %50 : f64
// CHECK-NEXT:        %55 = arith.addf %51, %52 : f64
// CHECK-NEXT:        %56 = arith.addf %54, %55 : f64
// CHECK-NEXT:        %cst_4 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %57 = arith.mulf %53, %cst_4 : f64
// CHECK-NEXT:        %58 = arith.addf %57, %56 : f64
// CHECK-NEXT:        %59 = stencil.access %8[-1, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %60 = stencil.access %8[1, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %61 = stencil.access %8[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %62 = stencil.access %8[0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %63 = stencil.access %8[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %64 = arith.addf %59, %60 : f64
// CHECK-NEXT:        %65 = arith.addf %61, %62 : f64
// CHECK-NEXT:        %66 = arith.addf %64, %65 : f64
// CHECK-NEXT:        %cst_5 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %67 = arith.mulf %63, %cst_5 : f64
// CHECK-NEXT:        %68 = arith.addf %67, %66 : f64
// CHECK-NEXT:        %69 = stencil.access %8[-1, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %70 = stencil.access %8[1, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %71 = stencil.access %8[0, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %72 = stencil.access %8[0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %73 = stencil.access %8[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %74 = arith.addf %69, %70 : f64
// CHECK-NEXT:        %75 = arith.addf %71, %72 : f64
// CHECK-NEXT:        %76 = arith.addf %74, %75 : f64
// CHECK-NEXT:        %cst_6 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %77 = arith.mulf %73, %cst_6 : f64
// CHECK-NEXT:        %78 = arith.addf %77, %76 : f64
// CHECK-NEXT:        %79 = stencil.access %8[-1, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %80 = stencil.access %8[1, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %81 = stencil.access %8[0, 8, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %82 = stencil.access %8[0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %83 = stencil.access %8[0, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %84 = arith.addf %79, %80 : f64
// CHECK-NEXT:        %85 = arith.addf %81, %82 : f64
// CHECK-NEXT:        %86 = arith.addf %84, %85 : f64
// CHECK-NEXT:        %cst_7 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %87 = arith.mulf %83, %cst_7 : f64
// CHECK-NEXT:        %88 = arith.addf %87, %86 : f64
// CHECK-NEXT:        stencil.return %18, %17, %28, %27, %38, %37, %48, %47, %58, %57, %68, %67, %78, %77, %88, %87 unroll <[1, 8, 1]> : f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %6 to %4(<[0, 0, 0], [64, 64, 64]>) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
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
