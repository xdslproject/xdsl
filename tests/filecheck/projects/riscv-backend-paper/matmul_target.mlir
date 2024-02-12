// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

riscv.assembly_section ".text" {
  riscv.directive ".globl" "matmul"
  riscv.directive ".p2align" "2"

// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
  riscv_func.func @matmul(
    %X : !riscv.reg<a0>,
    %Y : !riscv.reg<a1>,
    %G : !riscv.reg<a2>
  ) {
    %X_moved = riscv.mv %X : (!riscv.reg<a0>) -> !riscv.reg<>
    %Y_moved = riscv.mv %Y : (!riscv.reg<a1>) -> !riscv.reg<>
    %G_moved = riscv.mv %G : (!riscv.reg<a2>) -> !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c512 = riscv.li 512 : () -> !riscv.reg<>

    %stride_pattern_0 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<8>, #builtin.int<8>], "strides" = [#builtin.int<8>, #builtin.int<0>, #builtin.int<64>], "dm" = #builtin.int<0>} : () -> !snitch_stream.stride_pattern_type<3>

    %stride_pattern_1 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<8>, #builtin.int<8>], "strides" = [#builtin.int<64>, #builtin.int<8>, #builtin.int<0>], "dm" = #builtin.int<1>} : () -> !snitch_stream.stride_pattern_type<3>

    "snitch_stream.streaming_region"(%X_moved, %Y_moved, %stride_pattern_0, %stride_pattern_1) <{"operandSegmentSizes" = array<i32: 2, 0, 2>}> ({
    ^bb0(%X_stream : !stream.readable<!riscv.freg<ft0>>, %Y_stream : !stream.readable<!riscv.freg<ft1>>):
      riscv_scf.for %g_i : !riscv.reg<> = %c0 to %c512 step %c8 {
        %G_dest = riscv.add %G_moved, %g_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %init = riscv.fld %G_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %g = riscv_scf.for %i : !riscv.reg<> = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<ft0>
          %y = riscv_snitch.read from %Y_stream : !riscv.freg<ft1>
          %res = riscv.fmadd.d %x, %y, %acc : (!riscv.freg<ft0>, !riscv.freg<ft1>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %G_dest, %g, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }
    }) : (!riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<3>, !snitch_stream.stride_pattern_type<3>) -> ()

    riscv_func.return
  }
}


// CHECK:       .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      mv t5, a0
// CHECK-NEXT:      mv t4, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t2, 512
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      li a3, 7
// CHECK-NEXT:      li a4, 7
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      scfgwi a3, 64
// CHECK-NEXT:      scfgwi a4, 96
// CHECK-NEXT:      scfgwi a5, 128
// CHECK-NEXT:      scfgwi t6, 192
// CHECK-NEXT:      li t6, -56
// CHECK-NEXT:      scfgwi t6, 224
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      scfgwi t6, 256
// CHECK-NEXT:      li t6, 64
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      li a4, 7
// CHECK-NEXT:      li a3, 7
// CHECK-NEXT:      scfgwi a5, 65
// CHECK-NEXT:      scfgwi a4, 97
// CHECK-NEXT:      scfgwi a3, 129
// CHECK-NEXT:      scfgwi t6, 193
// CHECK-NEXT:      li t6, -440
// CHECK-NEXT:      scfgwi t6, 225
// CHECK-NEXT:      li t6, -504
// CHECK-NEXT:      scfgwi t6, 257
// CHECK-NEXT:      scfgwi t5, 832
// CHECK-NEXT:      scfgwi t4, 833
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_0_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft3, 0(t4)
// CHECK-NEXT:      li t5, 7
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_0_for
// CHECK-NEXT:  scf_body_end_0_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
