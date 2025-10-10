// RUN: xdsl-opt -p test-vectorize-matmul,convert-vector-to-ptr,convert-memref-to-ptr{lower_func=true},convert-ptr-type-offsets,canonicalize,convert-func-to-x86-func,convert-vector-to-x86{arch=avx2},convert-ptr-to-x86{arch=avx2},convert-arith-to-x86,reconcile-unrealized-casts,canonicalize,x86-infer-broadcast,dce,x86-allocate-registers,canonicalize -t x86-asm %s | filecheck %s

func.func public @matmul(
  %A: memref<2x4xf64>,
  %B: memref<4x4xf64>,
  %C: memref<2x4xf64>
) {
  linalg.matmul ins(%A, %B: memref<2x4xf64>, memref<4x4xf64>) outs(%C: memref<2x4xf64>)
  return
}
// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      mov rcx, rdi
// CHECK-NEXT:      mov rbx, rsi
// CHECK-NEXT:      mov rax, rdx
// CHECK-NEXT:      vmovupd ymm1, [rax]
// CHECK-NEXT:      vmovupd ymm0, [rax+32]
// CHECK-NEXT:      vmovupd ymm9, [rbx]
// CHECK-NEXT:      vmovupd ymm7, [rbx+32]
// CHECK-NEXT:      vmovupd ymm5, [rbx+64]
// CHECK-NEXT:      vmovupd ymm3, [rbx+96]
// CHECK-NEXT:      vbroadcastsd ymm2, [rcx]
// CHECK-NEXT:      vbroadcastsd ymm4, [rcx+8]
// CHECK-NEXT:      vbroadcastsd ymm6, [rcx+16]
// CHECK-NEXT:      vbroadcastsd ymm8, [rcx+24]
// CHECK-NEXT:      vfmadd231pd ymm1, ymm2, ymm9
// CHECK-NEXT:      vfmadd231pd ymm1, ymm4, ymm7
// CHECK-NEXT:      vfmadd231pd ymm1, ymm6, ymm5
// CHECK-NEXT:      vfmadd231pd ymm1, ymm8, ymm3
// CHECK-NEXT:      vbroadcastsd ymm8, [rcx+32]
// CHECK-NEXT:      vbroadcastsd ymm6, [rcx+40]
// CHECK-NEXT:      vbroadcastsd ymm4, [rcx+48]
// CHECK-NEXT:      vbroadcastsd ymm2, [rcx+56]
// CHECK-NEXT:      vfmadd231pd ymm0, ymm8, ymm9
// CHECK-NEXT:      vfmadd231pd ymm0, ymm6, ymm7
// CHECK-NEXT:      vfmadd231pd ymm0, ymm4, ymm5
// CHECK-NEXT:      vfmadd231pd ymm0, ymm2, ymm3
// CHECK-NEXT:      vmovapd [rax], ymm1
// CHECK-NEXT:      vmovapd [rax+32], ymm0
// CHECK-NEXT:      ret
