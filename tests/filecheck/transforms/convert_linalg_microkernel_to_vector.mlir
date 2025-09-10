// RUN: xdsl-opt -p convert-linalg-microkernel-to-vector --verify-diagnostics --split-input-file %s | filecheck %s

%A = "test.op"(): () -> memref<2x4xf64>
%B = "test.op"(): () -> memref<4x4xf64>
%C = "test.op"(): () -> memref<2x4xf64>

linalg.matmul ins(%A, %B : memref<2x4xf64>, memref<4x4xf64>) outs(%C : memref<2x4xf64>)
// CHECK:       builtin.module {
// CHECK-NEXT:    %A = "test.op"() : () -> memref<2x4xf64>
// CHECK-NEXT:    %B = "test.op"() : () -> memref<4x4xf64>
// CHECK-NEXT:    %C = "test.op"() : () -> memref<2x4xf64>
// CHECK-NEXT:    %0 = arith.constant 0 : index
// CHECK-NEXT:    %1 = arith.constant 1 : index
// CHECK-NEXT:    %2 = arith.constant 0 : index
// CHECK-NEXT:    %3 = arith.constant 1 : index
// CHECK-NEXT:    %4 = arith.constant 2 : index
// CHECK-NEXT:    %5 = arith.constant 3 : index
// CHECK-NEXT:    %6 = arith.constant 0 : index
// CHECK-NEXT:    %7 = arith.constant 1 : index
// CHECK-NEXT:    %8 = arith.constant 2 : index
// CHECK-NEXT:    %9 = arith.constant 3 : index
// CHECK-NEXT:    %10 = vector.load %C[%0, %2] : memref<2x4xf64>, vector<4xf64>
// CHECK-NEXT:    %11 = vector.load %C[%1, %2] : memref<2x4xf64>, vector<4xf64>
// CHECK-NEXT:    %12 = vector.load %B[%6, %2] : memref<4x4xf64>, vector<4xf64>
// CHECK-NEXT:    %13 = vector.load %B[%7, %2] : memref<4x4xf64>, vector<4xf64>
// CHECK-NEXT:    %14 = vector.load %B[%8, %2] : memref<4x4xf64>, vector<4xf64>
// CHECK-NEXT:    %15 = vector.load %B[%9, %2] : memref<4x4xf64>, vector<4xf64>
// CHECK-NEXT:    %16 = memref.load %A[%0, %6] : memref<2x4xf64>
// CHECK-NEXT:    %17 = memref.load %A[%0, %7] : memref<2x4xf64>
// CHECK-NEXT:    %18 = memref.load %A[%0, %8] : memref<2x4xf64>
// CHECK-NEXT:    %19 = memref.load %A[%0, %9] : memref<2x4xf64>
// CHECK-NEXT:    %20 = vector.broadcast %16 : f64 to vector<4xf64>
// CHECK-NEXT:    %21 = vector.broadcast %17 : f64 to vector<4xf64>
// CHECK-NEXT:    %22 = vector.broadcast %18 : f64 to vector<4xf64>
// CHECK-NEXT:    %23 = vector.broadcast %19 : f64 to vector<4xf64>
// CHECK-NEXT:    %24 = vector.fma %20, %12, %10 : vector<4xf64>
// CHECK-NEXT:    %25 = vector.fma %21, %13, %24 : vector<4xf64>
// CHECK-NEXT:    %26 = vector.fma %22, %14, %25 : vector<4xf64>
// CHECK-NEXT:    %27 = vector.fma %23, %15, %26 : vector<4xf64>
// CHECK-NEXT:    %28 = memref.load %A[%1, %6] : memref<2x4xf64>
// CHECK-NEXT:    %29 = memref.load %A[%1, %7] : memref<2x4xf64>
// CHECK-NEXT:    %30 = memref.load %A[%1, %8] : memref<2x4xf64>
// CHECK-NEXT:    %31 = memref.load %A[%1, %9] : memref<2x4xf64>
// CHECK-NEXT:    %32 = vector.broadcast %28 : f64 to vector<4xf64>
// CHECK-NEXT:    %33 = vector.broadcast %29 : f64 to vector<4xf64>
// CHECK-NEXT:    %34 = vector.broadcast %30 : f64 to vector<4xf64>
// CHECK-NEXT:    %35 = vector.broadcast %31 : f64 to vector<4xf64>
// CHECK-NEXT:    %36 = vector.fma %32, %12, %11 : vector<4xf64>
// CHECK-NEXT:    %37 = vector.fma %33, %13, %36 : vector<4xf64>
// CHECK-NEXT:    %38 = vector.fma %34, %14, %37 : vector<4xf64>
// CHECK-NEXT:    %39 = vector.fma %35, %15, %38 : vector<4xf64>
// CHECK-NEXT:    vector.store %27, %C[%0, %2] : memref<2x4xf64>, vector<4xf64>
// CHECK-NEXT:    vector.store %39, %C[%1, %2] : memref<2x4xf64>, vector<4xf64>
// CHECK-NEXT:  }

// -----

// CHECK: linalg.matmul on tensors is not yet implemented

%A = "test.op"(): () -> tensor<2x4xf64>
%B = "test.op"(): () -> tensor<4x4xf64>
%C = "test.op"(): () -> tensor<2x4xf64>

linalg.matmul ins(%A, %B : tensor<2x4xf64>, tensor<4x4xf64>) outs(%C : tensor<2x4xf64>)

// -----

// CHECK: MemRefs with ranks higher than 2 are not supported

%A = "test.op"(): () -> memref<2x4x1xf64>
%B = "test.op"(): () -> memref<4x1x4xf64>
%C = "test.op"(): () -> memref<2x4xf64>

linalg.matmul ins(%A, %B : memref<2x4x1xf64>, memref<4x1x4xf64>) outs(%C : memref<2x4xf64>)
