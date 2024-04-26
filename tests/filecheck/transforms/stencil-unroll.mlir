// RUN: xdsl-opt %s -p "stencil-unroll{unroll-factor=8,1}" | filecheck %s

func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = "stencil.apply"(%0) ({
    ^0(%4 : f64):
      %5 = arith.constant 1.0 : f64
      %6 = arith.addf %4, %5 : f64
      "stencil.return"(%6) : (f64) -> ()
    }) : (f64) -> !stencil.temp<[1,65]x[2,66]x[3,63]xf64>
    "stencil.store"(%3, %2) {"lb" = #stencil.index[1, 2, 3], "ub" = #stencil.index[65, 66, 63]} : (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>, !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) -> ()
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

  func.func @copy_1d(%0 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
    %1 = "stencil.cast"(%0) : (!stencil.field<?xf64>) -> !stencil.field<[-4,68]xf64>
    %outc = "stencil.cast"(%out) : (!stencil.field<?xf64>) -> !stencil.field<[0,1024]xf64>
    %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
    %3 = "stencil.apply"(%2) ({
    ^0(%4 : !stencil.temp<[-1,68]xf64>):
      %5 = "stencil.access"(%4) {"offset" = #stencil.index[-1]} : (!stencil.temp<[-1,68]xf64>) -> f64
      "stencil.return"(%5) : (f64) -> ()
    }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "stencil.store"(%3, %outc) {"lb" = #stencil.index[0], "ub" = #stencil.index[68]} : (!stencil.temp<[0,68]xf64>, !stencil.field<[0,1024]xf64>) -> ()
    func.return
  }

// CHECK:         func.func @copy_1d(%21 : !stencil.field<?xf64>, %out : !stencil.field<?xf64>) {
// CHECK-NEXT:      %22 = stencil.cast %21 : !stencil.field<?xf64> -> !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      %outc = stencil.cast %out : !stencil.field<?xf64> -> !stencil.field<[0,1024]xf64>
// CHECK-NEXT:      %23 = stencil.load %22 : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,68]xf64>
// CHECK-NEXT:      %24 = stencil.apply(%25 = %23 : !stencil.temp<[-1,68]xf64>) -> (!stencil.temp<[0,68]xf64>) {
// CHECK-NEXT:        %26 = stencil.access %25 [-1] : !stencil.temp<[-1,68]xf64>
// CHECK-NEXT:        stencil.return %26 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %24 to %outc ([0] : [68]) : !stencil.temp<[0,68]xf64> to !stencil.field<[0,1024]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @offsets(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>, %2 : !stencil.field<?x?x?xf64>) {
    %3 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %5 = "stencil.cast"(%2) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %6 = "stencil.load"(%3) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
    %7, %8 = "stencil.apply"(%6) ({
    ^0(%9 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index[-1, 0, 0]} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index[1, 0, 0]} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index[0, 1, 0]} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index[0, -1, 0]} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index[0, 0, 0]} : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> f64
      %15 = arith.addf %10, %11 : f64
      %16 = arith.addf %12, %13 : f64
      %17 = arith.addf %15, %16 : f64
      %cst = arith.constant -4.0 : f64
      %18 = arith.mulf %14, %cst : f64
      %19 = arith.addf %18, %17 : f64
      "stencil.return"(%19, %18) : (f64, f64) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>)
    "stencil.store"(%7, %4) {"lb" = #stencil.index[0, 0, 0], "ub" = #stencil.index[64, 64, 64]} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    func.return
  }

// CHECK:         func.func @offsets(%27 : !stencil.field<?x?x?xf64>, %28 : !stencil.field<?x?x?xf64>, %29 : !stencil.field<?x?x?xf64>) {
// CHECK-NEXT:      %30 = stencil.cast %27 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %31 = stencil.cast %28 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %32 = stencil.cast %29 : !stencil.field<?x?x?xf64> -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      %33 = stencil.load %30 : !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:      %34, %35 = stencil.apply(%36 = %33 : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.temp<[0,64]x[0,64]x[0,64]xf64>) {
// CHECK-NEXT:        %37 = stencil.access %36 [-1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %38 = stencil.access %36 [1, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %39 = stencil.access %36 [0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %40 = stencil.access %36 [0, -1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %41 = stencil.access %36 [0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %42 = arith.addf %37, %38 : f64
// CHECK-NEXT:        %43 = arith.addf %39, %40 : f64
// CHECK-NEXT:        %44 = arith.addf %42, %43 : f64
// CHECK-NEXT:        %45 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %46 = arith.mulf %41, %45 : f64
// CHECK-NEXT:        %47 = arith.addf %46, %44 : f64
// CHECK-NEXT:        %48 = stencil.access %36 [-1, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %49 = stencil.access %36 [1, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %50 = stencil.access %36 [0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %51 = stencil.access %36 [0, 0, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %52 = stencil.access %36 [0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %53 = arith.addf %48, %49 : f64
// CHECK-NEXT:        %54 = arith.addf %50, %51 : f64
// CHECK-NEXT:        %55 = arith.addf %53, %54 : f64
// CHECK-NEXT:        %56 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %57 = arith.mulf %52, %56 : f64
// CHECK-NEXT:        %58 = arith.addf %57, %55 : f64
// CHECK-NEXT:        %59 = stencil.access %36 [-1, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %60 = stencil.access %36 [1, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %61 = stencil.access %36 [0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %62 = stencil.access %36 [0, 1, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %63 = stencil.access %36 [0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %64 = arith.addf %59, %60 : f64
// CHECK-NEXT:        %65 = arith.addf %61, %62 : f64
// CHECK-NEXT:        %66 = arith.addf %64, %65 : f64
// CHECK-NEXT:        %67 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %68 = arith.mulf %63, %67 : f64
// CHECK-NEXT:        %69 = arith.addf %68, %66 : f64
// CHECK-NEXT:        %70 = stencil.access %36 [-1, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %71 = stencil.access %36 [1, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %72 = stencil.access %36 [0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %73 = stencil.access %36 [0, 2, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %74 = stencil.access %36 [0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %75 = arith.addf %70, %71 : f64
// CHECK-NEXT:        %76 = arith.addf %72, %73 : f64
// CHECK-NEXT:        %77 = arith.addf %75, %76 : f64
// CHECK-NEXT:        %78 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %79 = arith.mulf %74, %78 : f64
// CHECK-NEXT:        %80 = arith.addf %79, %77 : f64
// CHECK-NEXT:        %81 = stencil.access %36 [-1, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %82 = stencil.access %36 [1, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %83 = stencil.access %36 [0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %84 = stencil.access %36 [0, 3, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %85 = stencil.access %36 [0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %86 = arith.addf %81, %82 : f64
// CHECK-NEXT:        %87 = arith.addf %83, %84 : f64
// CHECK-NEXT:        %88 = arith.addf %86, %87 : f64
// CHECK-NEXT:        %89 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %90 = arith.mulf %85, %89 : f64
// CHECK-NEXT:        %91 = arith.addf %90, %88 : f64
// CHECK-NEXT:        %92 = stencil.access %36 [-1, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %93 = stencil.access %36 [1, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %94 = stencil.access %36 [0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %95 = stencil.access %36 [0, 4, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %96 = stencil.access %36 [0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %97 = arith.addf %92, %93 : f64
// CHECK-NEXT:        %98 = arith.addf %94, %95 : f64
// CHECK-NEXT:        %99 = arith.addf %97, %98 : f64
// CHECK-NEXT:        %100 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %101 = arith.mulf %96, %100 : f64
// CHECK-NEXT:        %102 = arith.addf %101, %99 : f64
// CHECK-NEXT:        %103 = stencil.access %36 [-1, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %104 = stencil.access %36 [1, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %105 = stencil.access %36 [0, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %106 = stencil.access %36 [0, 5, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %107 = stencil.access %36 [0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %108 = arith.addf %103, %104 : f64
// CHECK-NEXT:        %109 = arith.addf %105, %106 : f64
// CHECK-NEXT:        %110 = arith.addf %108, %109 : f64
// CHECK-NEXT:        %111 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %112 = arith.mulf %107, %111 : f64
// CHECK-NEXT:        %113 = arith.addf %112, %110 : f64
// CHECK-NEXT:        %114 = stencil.access %36 [-1, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %115 = stencil.access %36 [1, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %116 = stencil.access %36 [0, 8, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %117 = stencil.access %36 [0, 6, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %118 = stencil.access %36 [0, 7, 0] : !stencil.temp<[-1,65]x[-1,65]x[0,64]xf64>
// CHECK-NEXT:        %119 = arith.addf %114, %115 : f64
// CHECK-NEXT:        %120 = arith.addf %116, %117 : f64
// CHECK-NEXT:        %121 = arith.addf %119, %120 : f64
// CHECK-NEXT:        %122 = arith.constant -4.000000e+00 : f64
// CHECK-NEXT:        %123 = arith.mulf %118, %122 : f64
// CHECK-NEXT:        %124 = arith.addf %123, %121 : f64
// CHECK-NEXT:        stencil.return %47, %46, %58, %57, %69, %68, %80, %79, %91, %90, %102, %101, %113, %112, %124, %123 unroll [1, 8, 1] : f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %34 to %31 ([0, 0, 0] : [64, 64, 64]) : !stencil.temp<[0,64]x[0,64]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes { stencil.program } {
  %0 = "stencil.cast"(%arg0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %1 = "stencil.cast"(%arg1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %2 = "stencil.load"(%0) : (!stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
  %3 = "stencil.apply"(%2) ( {
  ^bb0(%arg2: !stencil.temp<[0,64]x[0,64]x[0,60]xf64>):  // no predecessors
    %4 = "stencil.index"() {dim = 0 : index, offset = #stencil.index[0, 0, 0]} : () -> index
    %5 = "stencil.index"() {dim = 1 : index, offset = #stencil.index[0, 0, 0]} : () -> index
    %6 = "stencil.index"() {dim = 2 : index, offset = #stencil.index[0, 0, 0]} : () -> index
    %7 = "stencil.dyn_access"(%arg2, %4, %5, %6) {lb = #stencil.index[0, 0, 0], ub = #stencil.index[0, 0, 0]} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, index, index, index) -> f64
    %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    "stencil.return"(%8) : (!stencil.result<f64>) -> ()
  }) {lb = #stencil.index[0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
  "stencil.store"(%3, %1) {lb = #stencil.index[0, 0, 0], ub = #stencil.index[64, 64, 60]} : (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>, !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> ()
  return
}

// CHECK:         func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>)  attributes {"stencil.program"}{
// CHECK-NEXT:      %125 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %126 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      %127 = stencil.load %125 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:      %128 = stencil.apply(%129 = %127 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
// CHECK-NEXT:        %130 = stencil.index 0 [0, 0, 0]
// CHECK-NEXT:        %131 = stencil.index 1 [0, 0, 0]
// CHECK-NEXT:        %132 = stencil.index 2 [0, 0, 0]
// CHECK-NEXT:        %133 = stencil.dyn_access %129[%130, %131, %132] in [0, 0, 0] : [0, 0, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %134 = stencil.store_result %133 : !stencil.result<f64>
// CHECK-NEXT:        %135 = stencil.index 0 [0, 1, 0]
// CHECK-NEXT:        %136 = stencil.index 1 [0, 1, 0]
// CHECK-NEXT:        %137 = stencil.index 2 [0, 1, 0]
// CHECK-NEXT:        %138 = stencil.dyn_access %129[%135, %136, %137] in [0, 1, 0] : [0, 1, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %139 = stencil.store_result %138 : !stencil.result<f64>
// CHECK-NEXT:        %140 = stencil.index 0 [0, 2, 0]
// CHECK-NEXT:        %141 = stencil.index 1 [0, 2, 0]
// CHECK-NEXT:        %142 = stencil.index 2 [0, 2, 0]
// CHECK-NEXT:        %143 = stencil.dyn_access %129[%140, %141, %142] in [0, 2, 0] : [0, 2, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %144 = stencil.store_result %143 : !stencil.result<f64>
// CHECK-NEXT:        %145 = stencil.index 0 [0, 3, 0]
// CHECK-NEXT:        %146 = stencil.index 1 [0, 3, 0]
// CHECK-NEXT:        %147 = stencil.index 2 [0, 3, 0]
// CHECK-NEXT:        %148 = stencil.dyn_access %129[%145, %146, %147] in [0, 3, 0] : [0, 3, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %149 = stencil.store_result %148 : !stencil.result<f64>
// CHECK-NEXT:        %150 = stencil.index 0 [0, 4, 0]
// CHECK-NEXT:        %151 = stencil.index 1 [0, 4, 0]
// CHECK-NEXT:        %152 = stencil.index 2 [0, 4, 0]
// CHECK-NEXT:        %153 = stencil.dyn_access %129[%150, %151, %152] in [0, 4, 0] : [0, 4, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %154 = stencil.store_result %153 : !stencil.result<f64>
// CHECK-NEXT:        %155 = stencil.index 0 [0, 5, 0]
// CHECK-NEXT:        %156 = stencil.index 1 [0, 5, 0]
// CHECK-NEXT:        %157 = stencil.index 2 [0, 5, 0]
// CHECK-NEXT:        %158 = stencil.dyn_access %129[%155, %156, %157] in [0, 5, 0] : [0, 5, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %159 = stencil.store_result %158 : !stencil.result<f64>
// CHECK-NEXT:        %160 = stencil.index 0 [0, 6, 0]
// CHECK-NEXT:        %161 = stencil.index 1 [0, 6, 0]
// CHECK-NEXT:        %162 = stencil.index 2 [0, 6, 0]
// CHECK-NEXT:        %163 = stencil.dyn_access %129[%160, %161, %162] in [0, 6, 0] : [0, 6, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %164 = stencil.store_result %163 : !stencil.result<f64>
// CHECK-NEXT:        %165 = stencil.index 0 [0, 7, 0]
// CHECK-NEXT:        %166 = stencil.index 1 [0, 7, 0]
// CHECK-NEXT:        %167 = stencil.index 2 [0, 7, 0]
// CHECK-NEXT:        %168 = stencil.dyn_access %129[%165, %166, %167] in [0, 7, 0] : [0, 7, 0] : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
// CHECK-NEXT:        %169 = stencil.store_result %168 : !stencil.result<f64>
// CHECK-NEXT:        stencil.return %134, %139, %144, %149, %154, %159, %164, %169 unroll [1, 8, 1] : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %128 to %126 ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
