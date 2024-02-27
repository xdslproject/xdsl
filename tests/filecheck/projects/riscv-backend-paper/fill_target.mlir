// RUN: xdsl-opt -p convert-func-to-riscv-func,reconcile-unrealized-casts,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s

  // y[ 16 x 16 ]
  func.func @fill(
    %X : f64,
    %Y : memref<16x16xf64>
  ) {
    %X_moved = builtin.unrealized_conversion_cast %X : f64 to !riscv.freg<>
    %Y_moved = builtin.unrealized_conversion_cast %Y : memref<16x16xf64> to !riscv.reg<>

    %x = riscv.fmv.d %X_moved : (!riscv.freg<>) -> !riscv.freg<>

    "snitch_stream.streaming_region"(%Y_moved) <{
      "stride_patterns" = [#snitch_stream.stride_pattern<ub = [256], strides=[8]>],
      "operandSegmentSizes" = array<i32: 0, 1>
    }> ({
    ^bb0(%Y_stream : !stream.writable<!riscv.freg<ft0>>):
      %c0 = riscv.li 0 : () -> !riscv.reg<>
      %c1 = riscv.li 1 : () -> !riscv.reg<>
      %c256 = riscv.li 256 : () -> !riscv.reg<>
      riscv_scf.for %i : !riscv.reg<> = %c0 to %c256 step %c1 {
        %y = riscv.fmv.d %x : (!riscv.freg<>) -> !riscv.freg<ft0>
        riscv_snitch.write %y to %Y_stream : !riscv.freg<ft0>
      }
    }) : (!riscv.reg<>) -> ()

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl fill
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  fill:
// CHECK-NEXT:      fmv.d ft3, fa0
// CHECK-NEXT:      mv t0, a0
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      li t2, 255
// CHECK-NEXT:      scfgwi t2, 64
// CHECK-NEXT:      scfgwi t1, 192
// CHECK-NEXT:      scfgwi t0, 896
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t0, 255
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fmv.d ft0, ft3
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
