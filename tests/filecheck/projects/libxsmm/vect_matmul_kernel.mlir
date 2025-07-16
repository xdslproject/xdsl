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

// CHECK:       matmul:
// CHECK-NEXT:      mov rcx, rdi
// CHECK-NEXT:      mov rbx, rsi
// CHECK-NEXT:      mov rax, rdx
// CHECK-NEXT:      mov r8, rax
// CHECK-NEXT:      vmovupd ymm1, [r8]
// CHECK-NEXT:      mov r8, 32
// CHECK-NEXT:      mov r11, rax
// CHECK-NEXT:      add r11, r8
// CHECK-NEXT:      vmovupd ymm0, [r11]
// CHECK-NEXT:      mov r11, rbx
// CHECK-NEXT:      vmovupd ymm9, [r11]
// CHECK-NEXT:      mov r11, 32
// CHECK-NEXT:      mov r8, rbx
// CHECK-NEXT:      add r8, r11
// CHECK-NEXT:      vmovupd ymm7, [r8]
// CHECK-NEXT:      mov r8, 64
// CHECK-NEXT:      mov r11, rbx
// CHECK-NEXT:      add r11, r8
// CHECK-NEXT:      vmovupd ymm5, [r11]
// CHECK-NEXT:      mov r11, 96
// CHECK-NEXT:      add rbx, r11
// CHECK-NEXT:      vmovupd ymm3, [rbx]
// CHECK-NEXT:      mov rbx, rcx
// CHECK-NEXT:      mov rbx, [rbx]
// CHECK-NEXT:      mov r11, 8
// CHECK-NEXT:      mov r8, rcx
// CHECK-NEXT:      add r8, r11
// CHECK-NEXT:      mov r8, [r8]
// CHECK-NEXT:      mov r11, 16
// CHECK-NEXT:      mov r9, rcx
// CHECK-NEXT:      add r9, r11
// CHECK-NEXT:      mov r9, [r9]
// CHECK-NEXT:      mov r11, 24
// CHECK-NEXT:      mov r10, rcx
// CHECK-NEXT:      add r10, r11
// CHECK-NEXT:      mov r10, [r10]
// CHECK-NEXT:      vpbroadcastq ymm2, rbx
// CHECK-NEXT:      vpbroadcastq ymm4, r8
// CHECK-NEXT:      vpbroadcastq ymm6, r9
// CHECK-NEXT:      vpbroadcastq ymm8, r10
// CHECK-NEXT:      vfmadd231pd ymm1, ymm2, ymm9
// CHECK-NEXT:      vfmadd231pd ymm1, ymm4, ymm7
// CHECK-NEXT:      vfmadd231pd ymm1, ymm6, ymm5
// CHECK-NEXT:      vfmadd231pd ymm1, ymm8, ymm3
// CHECK-NEXT:      mov r10, 32
// CHECK-NEXT:      mov r9, rcx
// CHECK-NEXT:      add r9, r10
// CHECK-NEXT:      mov r9, [r9]
// CHECK-NEXT:      mov r10, 40
// CHECK-NEXT:      mov r8, rcx
// CHECK-NEXT:      add r8, r10
// CHECK-NEXT:      mov r8, [r8]
// CHECK-NEXT:      mov r10, 48
// CHECK-NEXT:      mov rbx, rcx
// CHECK-NEXT:      add rbx, r10
// CHECK-NEXT:      mov rbx, [rbx]
// CHECK-NEXT:      mov r10, 56
// CHECK-NEXT:      add rcx, r10
// CHECK-NEXT:      mov rcx, [rcx]
// CHECK-NEXT:      vpbroadcastq ymm8, r9
// CHECK-NEXT:      vpbroadcastq ymm6, r8
// CHECK-NEXT:      vpbroadcastq ymm4, rbx
// CHECK-NEXT:      vpbroadcastq ymm2, rcx
// CHECK-NEXT:      vfmadd231pd ymm0, ymm8, ymm9
// CHECK-NEXT:      vfmadd231pd ymm0, ymm6, ymm7
// CHECK-NEXT:      vfmadd231pd ymm0, ymm4, ymm5
// CHECK-NEXT:      vfmadd231pd ymm0, ymm2, ymm3
// CHECK-NEXT:      mov rcx, rax
// CHECK-NEXT:      vmovapd [rcx], ymm1
// CHECK-NEXT:      vmovapd [rax+32], ymm0
// CHECK-NEXT:      ret
