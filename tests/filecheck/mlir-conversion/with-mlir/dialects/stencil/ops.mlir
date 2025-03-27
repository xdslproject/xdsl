// RUN: xdsl-opt %s | xdsl-opt -p mlir-opt[math-uplift-to-fma] | filecheck %s
// note, this selected pass should have no effect

func.func @dyn_access(%arg0 : !stencil.field<?x?x?xf64>, %arg1 : !stencil.field<?x?x?xf64>) attributes {"stencil.program"} {
  %0 = stencil.cast %arg0 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %1 = stencil.cast %arg1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  %2 = stencil.load %0 : !stencil.field<[-3,67]x[-3,67]x[0,60]xf64> -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
  %3 = stencil.apply(%arg2 = %2 : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,60]xf64>) {
    %4 = stencil.index 0 <[0, 0, 0]>
    %5 = stencil.index 1 <[0, 0, 0]>
    %6 = stencil.index 2 <[0, 0, 0]>
    %7 = stencil.dyn_access %arg2[%4, %5, %6] in <[0, 0, 0]> : <[0, 0, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %8 = stencil.store_result %7 : !stencil.result<f64>
    %9 = stencil.index 0 <[0, 1, 0]>
    %10 = stencil.index 1 <[0, 1, 0]>
    %11 = stencil.index 2 <[0, 1, 0]>
    %12 = stencil.dyn_access %arg2[%9, %10, %11] in <[0, 1, 0]> : <[0, 1, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %13 = stencil.store_result %12 : !stencil.result<f64>
    %14 = stencil.index 0 <[0, 2, 0]>
    %15 = stencil.index 1 <[0, 2, 0]>
    %16 = stencil.index 2 <[0, 2, 0]>
    %17 = stencil.dyn_access %arg2[%14, %15, %16] in <[0, 2, 0]> : <[0, 2, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %18 = stencil.store_result %17 : !stencil.result<f64>
    %19 = stencil.index 0 <[0, 3, 0]>
    %20 = stencil.index 1 <[0, 3, 0]>
    %21 = stencil.index 2 <[0, 3, 0]>
    %22 = stencil.dyn_access %arg2[%19, %20, %21] in <[0, 3, 0]> : <[0, 3, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %23 = stencil.store_result %22 : !stencil.result<f64>
    %24 = stencil.index 0 <[0, 4, 0]>
    %25 = stencil.index 1 <[0, 4, 0]>
    %26 = stencil.index 2 <[0, 4, 0]>
    %27 = stencil.dyn_access %arg2[%24, %25, %26] in <[0, 4, 0]> : <[0, 4, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %28 = stencil.store_result %27 : !stencil.result<f64>
    %29 = stencil.index 0 <[0, 5, 0]>
    %30 = stencil.index 1 <[0, 5, 0]>
    %31 = stencil.index 2 <[0, 5, 0]>
    %32 = stencil.dyn_access %arg2[%29, %30, %31] in <[0, 5, 0]> : <[0, 5, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %33 = stencil.store_result %32 : !stencil.result<f64>
    %34 = stencil.index 0 <[0, 6, 0]>
    %35 = stencil.index 1 <[0, 6, 0]>
    %36 = stencil.index 2 <[0, 6, 0]>
    %37 = stencil.dyn_access %arg2[%34, %35, %36] in <[0, 6, 0]> : <[0, 6, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %38 = stencil.store_result %37 : !stencil.result<f64>
    %39 = stencil.index 0 <[0, 7, 0]>
    %40 = stencil.index 1 <[0, 7, 0]>
    %41 = stencil.index 2 <[0, 7, 0]>
    %42 = stencil.dyn_access %arg2[%39, %40, %41] in <[0, 7, 0]> : <[0, 7, 0]> : !stencil.temp<[0,64]x[0,64]x[0,60]xf64>
    %43 = stencil.store_result %42 : !stencil.result<f64>
    stencil.return %8, %13, %18, %23, %28, %33, %38, %43 unroll <[1, 8, 1]> : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
  }
  stencil.store %3 to %1(<[0, 0, 0], [64, 64, 60]>) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  func.return
}

// CHECK:      builtin.module {
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
// CHECK-NEXT:  }
