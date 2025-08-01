// RUN: xdsl-opt -p convert-vector-to-ptr,convert-memref-to-ptr{lower_func=true},convert-ptr-type-offsets,canonicalize,convert-func-to-x86-func,convert-vector-to-x86{arch=avx2},convert-ptr-to-x86{arch=avx2},convert-arith-to-x86,reconcile-unrealized-casts,canonicalize,x86-allocate-registers,canonicalize -t x86-asm %s | filecheck %s

func.func @matmul(
  %A: memref<2x4xf64>,
  %B: memref<4x4xf64>,
  %C: memref<2x4xf64>
) {
  %i0 = arith.constant 0: index
  %i1 = arith.constant 1: index
  %j0 = arith.constant 0: index
  %k0 = arith.constant 0: index
  %k1 = arith.constant 1: index
  %k2 = arith.constant 2: index
  %k3 = arith.constant 3: index

  // Load C lines
  %c_line0 = vector.load %C[%i0,%j0]: memref<2x4xf64>, vector<4xf64>
  %c_line1 = vector.load %C[%i1,%j0]: memref<2x4xf64>, vector<4xf64>
  // Load B lines
  %b_line0 = vector.load %B[%k0,%j0]: memref<4x4xf64>, vector<4xf64>
  %b_line1 = vector.load %B[%k1,%j0]: memref<4x4xf64>, vector<4xf64>
  %b_line2 = vector.load %B[%k2,%j0]: memref<4x4xf64>, vector<4xf64>
  %b_line3 = vector.load %B[%k3,%j0]: memref<4x4xf64>, vector<4xf64>

  // Load column 0 of A
  %a_00_scal = memref.load %A[%i0, %k0] : memref<2x4xf64>
  %a_01_scal = memref.load %A[%i0, %k1] : memref<2x4xf64>
  %a_02_scal = memref.load %A[%i0, %k2] : memref<2x4xf64>
  %a_03_scal = memref.load %A[%i0, %k3] : memref<2x4xf64>
  %a_00 = vector.broadcast %a_00_scal: f64 to vector<4xf64>
  %a_01 = vector.broadcast %a_01_scal: f64 to vector<4xf64>
  %a_02 = vector.broadcast %a_02_scal: f64 to vector<4xf64>
  %a_03 = vector.broadcast %a_03_scal: f64 to vector<4xf64>
  // Perform the reduction
  %c_line0_acc0 = vector.fma %a_00, %b_line0, %c_line0: vector<4xf64>
  %c_line0_acc1 = vector.fma %a_01, %b_line1, %c_line0_acc0: vector<4xf64>
  %c_line0_acc2 = vector.fma %a_02, %b_line2, %c_line0_acc1: vector<4xf64>
  %c_line0_acc3 = vector.fma %a_03, %b_line3, %c_line0_acc2: vector<4xf64>

  // Load column 1 of A
  %a_10_scal = memref.load %A[%i1, %k0] : memref<2x4xf64>
  %a_11_scal = memref.load %A[%i1, %k1] : memref<2x4xf64>
  %a_12_scal = memref.load %A[%i1, %k2] : memref<2x4xf64>
  %a_13_scal = memref.load %A[%i1, %k3] : memref<2x4xf64>
  %a_10 = vector.broadcast %a_10_scal: f64 to vector<4xf64>
  %a_11 = vector.broadcast %a_11_scal: f64 to vector<4xf64>
  %a_12 = vector.broadcast %a_12_scal: f64 to vector<4xf64>
  %a_13 = vector.broadcast %a_13_scal: f64 to vector<4xf64>
  // Perform the reduction
  %c_line1_acc4 = vector.fma %a_10, %b_line0, %c_line1: vector<4xf64>
  %c_line1_acc5 = vector.fma %a_11, %b_line1, %c_line1_acc4: vector<4xf64>
  %c_line1_acc6 = vector.fma %a_12, %b_line2, %c_line1_acc5: vector<4xf64>
  %c_line1_acc7 = vector.fma %a_13, %b_line3, %c_line1_acc6: vector<4xf64>

  vector.store %c_line0_acc3, %C[%i0,%j0]: memref<2x4xf64>, vector<4xf64>
  vector.store %c_line1_acc7, %C[%i1,%j0]: memref<2x4xf64>, vector<4xf64>

  return
}
