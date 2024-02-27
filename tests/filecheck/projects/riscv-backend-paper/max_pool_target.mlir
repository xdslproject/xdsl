// RUN: xdsl-opt -p convert-func-to-riscv-func,reconcile-unrealized-casts,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s

// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
func.func public @pooling_nchw_max_d1_s2_3x3(
    %X: memref<1x1x16x16xf64>,
    %Y: memref<1x1x7x7xf64>
) -> () {
    %X_moved = builtin.unrealized_conversion_cast %X : memref<1x1x16x16xf64> to !riscv.reg<>
    %Y_moved = builtin.unrealized_conversion_cast %Y : memref<1x1x7x7xf64> to !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c9 = riscv.li 9 : () -> !riscv.reg<>
    %c512 = riscv.li 512 : () -> !riscv.reg<>

    "snitch_stream.streaming_region"(%X_moved) <{
      "stride_patterns" = [#snitch_stream.stride_pattern<ub = [3, 3, 7, 7], strides = [8, 128, 16, 256]>],
      "operandSegmentSizes" = array<i32: 1, 0>
    }> ({
    ^bb0(%X_stream : !stream.readable<!riscv.freg<ft0>>, %Y_stream : !stream.readable<!riscv.freg<ft1>>):
      %c392 = riscv.li 392 : () -> !riscv.reg<>
      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c392 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %init = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %y = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<ft0>
          %res = riscv.fmax.d %x, %acc : (!riscv.freg<ft0>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %Y_dest, %y, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }
    }) : (!riscv.reg<>) -> ()

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl pooling_nchw_max_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_max_d1_s2_3x3:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      li t4, 8
// CHECK-NEXT:      li a3, 2
// CHECK-NEXT:      li a2, 2
// CHECK-NEXT:      li t6, 6
// CHECK-NEXT:      li t5, 6
// CHECK-NEXT:      scfgwi a3, 64
// CHECK-NEXT:      scfgwi a2, 96
// CHECK-NEXT:      scfgwi t6, 128
// CHECK-NEXT:      scfgwi t5, 160
// CHECK-NEXT:      scfgwi t4, 192
// CHECK-NEXT:      li t4, 112
// CHECK-NEXT:      scfgwi t4, 224
// CHECK-NEXT:      li t4, -256
// CHECK-NEXT:      scfgwi t4, 256
// CHECK-NEXT:      li t4, -112
// CHECK-NEXT:      scfgwi t4, 288
// CHECK-NEXT:      scfgwi t2, 864
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t2, 392
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_0_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft3, 0(t4)
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmax.d ft3, ft0, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_0_for
// CHECK-NEXT:  scf_body_end_0_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
