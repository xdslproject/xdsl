// RUN: xdsl-opt -p convert-func-to-riscv-func,reconcile-unrealized-casts,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s

// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
func.func public @conv_2d_nchw_fchw_d1_s1_3x3(
    %X: memref<1x1x8x8xf64>,
    %Y: memref<1x1x3x3xf64>,
    %Z: memref<1x1x6x6xf64>
) -> () {
    %X_moved = builtin.unrealized_conversion_cast %X : memref<1x1x8x8xf64> to !riscv.reg<>
    %Y_moved = builtin.unrealized_conversion_cast %Y : memref<1x1x3x3xf64> to !riscv.reg<>
    %Z_moved = builtin.unrealized_conversion_cast %Z : memref<1x1x6x6xf64> to !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c9 = riscv.li 9 : () -> !riscv.reg<>

    %zero_float = riscv.fcvt.d.w %c0 : (!riscv.reg<>) -> !riscv.freg<>

    "snitch_stream.streaming_region"(%X_moved, %Y_moved) <{
      "stride_patterns" = [
        #snitch_stream.stride_pattern<ub = [3, 3, 6, 6], strides = [8, 64, 8, 64]>,
        #snitch_stream.stride_pattern<ub = [3, 3, 6, 6], strides = [8, 24, 0, 0]>
      ],
      "operandSegmentSizes" = array<i32: 2, 0>
    }> ({
    ^bb0(%X_stream : !stream.readable<!riscv.freg<ft0>>, %Y_stream : !stream.readable<!riscv.freg<ft1>>):
      %c288 = riscv.li 288 : () -> !riscv.reg<>
      riscv_scf.for %z_i : !riscv.reg<> = %c0 to %c288 step %c8 {
        %Z_dest = riscv.add %Z_moved, %z_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %c = riscv.fld %Z_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %z = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %c) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<ft0>
          %y = riscv_snitch.read from %Y_stream : !riscv.freg<ft1>
          %res = riscv.fmadd.d %x, %y, %acc : (!riscv.freg<ft0>, !riscv.freg<ft1>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %Z_dest, %z, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }
    }) : (!riscv.reg<>, !riscv.reg<>) -> ()

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl conv_2d_nchw_fchw_d1_s1_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  conv_2d_nchw_fchw_d1_s1_3x3:
// CHECK-NEXT:      mv t4, a0
// CHECK-NEXT:      mv t2, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      li t6, 2
// CHECK-NEXT:      li a3, 2
// CHECK-NEXT:      li a4, 5
// CHECK-NEXT:      li a5, 5
// CHECK-NEXT:      scfgwi t6, 64
// CHECK-NEXT:      scfgwi a3, 96
// CHECK-NEXT:      scfgwi a4, 128
// CHECK-NEXT:      scfgwi a5, 160
// CHECK-NEXT:      scfgwi t5, 192
// CHECK-NEXT:      li t5, 48
// CHECK-NEXT:      scfgwi t5, 224
// CHECK-NEXT:      li t5, -136
// CHECK-NEXT:      scfgwi t5, 256
// CHECK-NEXT:      li t5, -120
// CHECK-NEXT:      scfgwi t5, 288
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      li a5, 2
// CHECK-NEXT:      li a4, 2
// CHECK-NEXT:      li a3, 5
// CHECK-NEXT:      li t6, 5
// CHECK-NEXT:      scfgwi a5, 65
// CHECK-NEXT:      scfgwi a4, 97
// CHECK-NEXT:      scfgwi a3, 129
// CHECK-NEXT:      scfgwi t6, 161
// CHECK-NEXT:      scfgwi t5, 193
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      scfgwi t5, 225
// CHECK-NEXT:      li t5, -64
// CHECK-NEXT:      scfgwi t5, 257
// CHECK-NEXT:      li t5, -64
// CHECK-NEXT:      scfgwi t5, 289
// CHECK-NEXT:      scfgwi t4, 864
// CHECK-NEXT:      scfgwi t2, 865
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t2, 288
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_0_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft3, 0(t4)
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_0_for
// CHECK-NEXT:  scf_body_end_0_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
