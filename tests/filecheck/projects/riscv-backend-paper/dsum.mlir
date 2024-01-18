// RUN: xdsl-opt %s -p convert-linalg-to-memref-stream,convert-memref-stream-to-snitch-stream,convert-func-to-riscv-func,convert-memref-to-riscv,convert-arith-to-riscv,convert-scf-to-riscv-scf,dce,reconcile-unrealized-casts,snitch-allocate-registers,convert-snitch-stream-to-snitch,lower-snitch,canonicalize,riscv-scf-loop-range-folding,canonicalize,riscv-reduce-register-pressure,riscv-allocate-registers,canonicalize,lower-riscv-func,lower-riscv-scf-to-labels -t riscv-asm | filecheck %s

func.func public @dsum(%X: memref<8x16xf64>,
                      %Y: memref<8x16xf64>,
                      %Z: memref<8x16xf64>) -> () {
  "linalg.generic"(%X, %Y, %Z) <{
    "indexing_maps" = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ], "iterator_types" = [
      #linalg.iterator_type<parallel>,
      #linalg.iterator_type<parallel>,
      #linalg.iterator_type<parallel>
    ],
    "operandSegmentSizes" = array<i32: 2, 1>
  }> ({
    ^bb0(%x : f64, %y : f64, %z : f64):
      %r0 = arith.addf %x, %y : f64
      "linalg.yield"(%r0) : (f64) -> ()
    }) : (memref<8x16xf64>, memref<8x16xf64>, memref<8x16xf64>) -> ()
  func.return
}

// CHECK:       .text
// CHECK-NEXT:  .globl dsum
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  dsum:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t3, 128
// CHECK-NEXT:      li t5, 7
// CHECK-NEXT:      li t4, 15
// CHECK-NEXT:      scfgwi t5, 95
// CHECK-NEXT:      scfgwi t4, 127
// CHECK-NEXT:      scfgwi t3, 223
// CHECK-NEXT:      li t3, -888
// CHECK-NEXT:      scfgwi t3, 255
// CHECK-NEXT:      scfgwi t2, 800
// CHECK-NEXT:      scfgwi t1, 801
// CHECK-NEXT:      scfgwi t0, 930
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t0, 127
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fadd.d ft2, ft0, ft1
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
