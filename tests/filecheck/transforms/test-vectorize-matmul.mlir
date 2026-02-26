// RUN: xdsl-opt -p test-vectorize-matmul %s | filecheck %s

func.func @matmul(
  %A: memref<2x3xf64>,
  %B: memref<3x4xf64>,
  %C: memref<2x4xf64>
) {
  linalg.matmul ins(%A, %B: memref<2x3xf64>, memref<3x4xf64>) outs(%C: memref<2x4xf64>)
  return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @matmul(%A : memref<2x3xf64>, %B : memref<3x4xf64>, %C : memref<2x4xf64>) {
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 1 : index
// CHECK-NEXT:      %2 = arith.constant 2 : index
// CHECK-NEXT:      %3 = arith.constant 3 : index
// CHECK-NEXT:      %4 = vector.load %C[%0, %0] : memref<2x4xf64>, vector<4xf64>
// CHECK-NEXT:      %5 = vector.load %C[%1, %0] : memref<2x4xf64>, vector<4xf64>
// CHECK-NEXT:      %6 = vector.load %B[%0, %0] : memref<3x4xf64>, vector<4xf64>
// CHECK-NEXT:      %7 = vector.load %B[%1, %0] : memref<3x4xf64>, vector<4xf64>
// CHECK-NEXT:      %8 = vector.load %B[%2, %0] : memref<3x4xf64>, vector<4xf64>
// CHECK-NEXT:      %9 = memref.load %A[%0, %0] : memref<2x3xf64>
// CHECK-NEXT:      %10 = memref.load %A[%0, %1] : memref<2x3xf64>
// CHECK-NEXT:      %11 = memref.load %A[%0, %2] : memref<2x3xf64>
// CHECK-NEXT:      %12 = vector.broadcast %9 : f64 to vector<4xf64>
// CHECK-NEXT:      %13 = vector.broadcast %10 : f64 to vector<4xf64>
// CHECK-NEXT:      %14 = vector.broadcast %11 : f64 to vector<4xf64>
// CHECK-NEXT:      %15 = vector.fma %12, %6, %4 : vector<4xf64>
// CHECK-NEXT:      %16 = vector.fma %13, %7, %15 : vector<4xf64>
// CHECK-NEXT:      %17 = vector.fma %14, %8, %16 : vector<4xf64>
// CHECK-NEXT:      %18 = memref.load %A[%1, %0] : memref<2x3xf64>
// CHECK-NEXT:      %19 = memref.load %A[%1, %1] : memref<2x3xf64>
// CHECK-NEXT:      %20 = memref.load %A[%1, %2] : memref<2x3xf64>
// CHECK-NEXT:      %21 = vector.broadcast %18 : f64 to vector<4xf64>
// CHECK-NEXT:      %22 = vector.broadcast %19 : f64 to vector<4xf64>
// CHECK-NEXT:      %23 = vector.broadcast %20 : f64 to vector<4xf64>
// CHECK-NEXT:      %24 = vector.fma %21, %6, %5 : vector<4xf64>
// CHECK-NEXT:      %25 = vector.fma %22, %7, %24 : vector<4xf64>
// CHECK-NEXT:      %26 = vector.fma %23, %8, %25 : vector<4xf64>
// CHECK-NEXT:      vector.store %17, %C[%0, %0] : memref<2x4xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %26, %C[%1, %0] : memref<2x4xf64>, vector<4xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
