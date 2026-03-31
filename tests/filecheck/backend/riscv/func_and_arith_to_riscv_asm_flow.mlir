// RUN: xdsl-opt %s -p convert-func-to-riscv-func,inline-snrt{cluster-num=2},convert-arith-to-riscv,reconcile-unrealized-casts,riscv-allocate-registers,riscv-lower-parallel-mov -t riscv-asm | filecheck %s


func.func @test(%dst: i32, %src: i32) -> i32 {

  %stride = arith.constant 128 : i32
  %size = arith.constant 512 : i32
  %repeat = arith.constant 64 : i32

  %tx_id = "snrt.dma_start_2d"(%dst, %src, %stride, %stride, %size, %repeat) : (i32, i32, i32, i32, i32, i32) -> i32

  func.return %tx_id : i32
}


// CHECK: .text
// CHECK-NEXT: .globl test
// CHECK-NEXT: .p2align 2
// CHECK-NEXT: test:
// CHECK-NEXT:     mv t2, a0
// CHECK-NEXT:     mv t3, a1
// CHECK-NEXT:     li t1, 128
// CHECK-NEXT:     li t0, 512
// CHECK-NEXT:     dmsrc t3, zero
// CHECK-NEXT:     dmdst t2, zero
// CHECK-NEXT:     dmstr t1, t1
// CHECK-NEXT:     dmrep t0
// CHECK-NEXT:     dmcpyi t0, t0, 2
// CHECK-NEXT:     mv a0, t0
// CHECK-NEXT:     ret
