// RUN: xdsl-opt -p convert-func-to-riscv-func,reconcile-unrealized-casts,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s

    func.func @dsum(
      %arg0 : memref<8x16xf64>,
      %arg1 : memref<8x16xf64>,
      %arg2 : memref<8x16xf64>
    ) -> memref<8x16xf64> {
      %0 = builtin.unrealized_conversion_cast %arg0 : memref<8x16xf64> to !riscv.reg<>
      %1 = builtin.unrealized_conversion_cast %arg1 : memref<8x16xf64> to !riscv.reg<>
      %2 = builtin.unrealized_conversion_cast %arg2 : memref<8x16xf64> to !riscv.reg<>

      %c0 = riscv.li 0 : () -> !riscv.reg<>
      %c1 = riscv.li 1 : () -> !riscv.reg<>
      %c128 = riscv.li 128 : () -> !riscv.reg<>
      "snitch_stream.streaming_region"(%0, %1, %2) <{
        "stride_patterns" = [#snitch_stream.stride_pattern<ub = [8, 16], strides = [128, 8]>],
        "operandSegmentSizes" = array<i32: 2, 1>
      }> ({
      ^0(%5 : !stream.readable<!riscv.freg<ft0>>, %6 : !stream.readable<!riscv.freg<ft1>>, %7 : !stream.writable<!riscv.freg<ft2>>):
        riscv_scf.for %i : !riscv.reg<> = %c0 to %c128 step %c1 {
          %9 = riscv_snitch.read from %5 : !riscv.freg<ft0>
          %10 = riscv_snitch.read from %6 : !riscv.freg<ft1>
          %11 = riscv.fadd.d %9, %10 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
          riscv_snitch.write %11 to %7 : !riscv.freg<ft2>
          riscv_scf.yield
        }
      }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()

      %res = builtin.unrealized_conversion_cast %2 : !riscv.reg<> to memref<8x16xf64>
      func.return %res : memref<8x16xf64>
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
// CHECK-NEXT:      li t1, 127
// CHECK-NEXT:      frep.o t1, 1, 0, 0
// CHECK-NEXT:      fadd.d ft2, ft0, ft1
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      mv a0, t0
// CHECK-NEXT:      ret
