// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

riscv.assembly_section ".text" {
  riscv.directive ".globl" "pooling_nchw_sum_d1_s2_3x3"
  riscv.directive ".p2align" "2"
// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
riscv_func.func public @pooling_nchw_sum_d1_s2_3x3(
    %X: !riscv.reg<a0>,
    %Y: !riscv.reg<a1>
) -> () {
    %X_moved = riscv.mv %X : (!riscv.reg<a0>) -> !riscv.reg<>
    %Y_moved = riscv.mv %Y : (!riscv.reg<a1>) -> !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c9 = riscv.li 9 : () -> !riscv.reg<>
    %c512 = riscv.li 512 : () -> !riscv.reg<>

    %stride_pattern_0 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<3>, #builtin.int<3>, #builtin.int<7>, #builtin.int<7>], "strides" = [#builtin.int<8>, #builtin.int<128>, #builtin.int<16>, #builtin.int<256>], "dm" = #builtin.int<0>} : () -> !snitch_stream.stride_pattern_type<4>

    "snitch_stream.streaming_region"(%X_moved, %stride_pattern_0) <{"operandSegmentSizes" = array<i32: 1, 0, 1>}> ({
    ^bb0(%X_stream : !stream.readable<!riscv.freg<ft0>>, %Y_stream : !stream.readable<!riscv.freg<ft1>>):
      %c392 = riscv.li 392 : () -> !riscv.reg<>
      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c392 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %init = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %y = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<ft0>
          %res = riscv.fadd.d %x, %acc : (!riscv.freg<ft0>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %Y_dest, %y, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }

    }) : (!riscv.reg<>, !snitch_stream.stride_pattern_type<4>) -> ()

    riscv_func.return
  }
}

// CHECK:       .text
// CHECK-NEXT:  .globl pooling_nchw_sum_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_sum_d1_s2_3x3:
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
// CHECK-NEXT:      fadd.d ft3, ft0, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_0_for
// CHECK-NEXT:  scf_body_end_0_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
