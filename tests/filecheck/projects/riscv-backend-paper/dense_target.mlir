// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

riscv.assembly_section ".text" {
  riscv.directive ".globl" "dense"
  riscv.directive ".p2align" "2"

  // * Inputs:  x[ M x K ]
  // * Weights: w[ K x N ]
  // * Biases:  b[ M x N ]
  // * Outputs: y[ M x N ]
  riscv_func.func @dense(
    %X : !riscv.reg<a0>,
    %W : !riscv.reg<a1>,
    %B : !riscv.reg<a2>,
    %Y : !riscv.reg<a3>
  ) {
    %X_moved = riscv.mv %X : (!riscv.reg<a0>) -> !riscv.reg<>
    %W_moved = riscv.mv %W : (!riscv.reg<a1>) -> !riscv.reg<>
    %B_moved = riscv.mv %B : (!riscv.reg<a2>) -> !riscv.reg<>
    %Y_moved = riscv.mv %Y : (!riscv.reg<a3>) -> !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c512 = riscv.li 512 : () -> !riscv.reg<>

    %zero_float = riscv.fcvt.d.w %c0 : (!riscv.reg<>) -> !riscv.freg<>

    %stride_pattern_0 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<8>, #builtin.int<8>], "strides" = [#builtin.int<8>, #builtin.int<0>, #builtin.int<64>], "dm" = #builtin.int<0>} : () -> !snitch_stream.stride_pattern_type<3>

    %stride_pattern_1 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<8>, #builtin.int<8>], "strides" = [#builtin.int<64>, #builtin.int<8>, #builtin.int<0>], "dm" = #builtin.int<1>} : () -> !snitch_stream.stride_pattern_type<3>

    %stride_pattern_2 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<8>], "strides" = [#builtin.int<8>, #builtin.int<64>], "dm" = #builtin.int<2>} : () -> !snitch_stream.stride_pattern_type<2>

    "snitch_stream.streaming_region"(%X_moved, %W_moved, %B_moved, %stride_pattern_0, %stride_pattern_1, %stride_pattern_2) <{"operandSegmentSizes" = array<i32: 3, 0, 3>}> ({
    ^bb0(%X_stream : !stream.readable<!riscv.freg<ft0>>, %W_stream : !stream.readable<!riscv.freg<ft1>>, %B_stream : !stream.readable<!riscv.freg<ft2>>):
      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c512 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %c = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %dot = riscv_scf.for %i : !riscv.reg<> = %c0 to %c8 step %c1 iter_args(%acc = %c) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<ft0>
          %w = riscv_snitch.read from %W_stream : !riscv.freg<ft1>
          %res = riscv.fmadd.d %x, %w, %acc : (!riscv.freg<ft0>, !riscv.freg<ft1>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        %b = riscv.get_float_register : () -> !riscv.freg<ft2>
        %y_0 = riscv.fadd.d %b, %dot : (!riscv.freg<ft2>, !riscv.freg<>) -> !riscv.freg<>
        %y_1 = riscv.fmax.d %y_0, %zero_float : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

        riscv.fsd %Y_dest, %y_1, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }
    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<3>, !snitch_stream.stride_pattern_type<3>, !snitch_stream.stride_pattern_type<2>) -> ()

    riscv_func.return
  }
}

// CHECK:       .text
// CHECK-NEXT:  .globl dense
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  dense:
// CHECK-NEXT:      mv t6, a0
// CHECK-NEXT:      mv t5, a1
// CHECK-NEXT:      mv t4, a2
// CHECK-NEXT:      mv t0, a3
// CHECK-NEXT:      li t2, 512
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li a4, 8
// CHECK-NEXT:      li a6, 7
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      li a7, 7
// CHECK-NEXT:      scfgwi a6, 64
// CHECK-NEXT:      scfgwi a5, 96
// CHECK-NEXT:      scfgwi a7, 128
// CHECK-NEXT:      scfgwi a4, 192
// CHECK-NEXT:      li a4, -56
// CHECK-NEXT:      scfgwi a4, 224
// CHECK-NEXT:      li a4, 8
// CHECK-NEXT:      scfgwi a4, 256
// CHECK-NEXT:      li a4, 64
// CHECK-NEXT:      li a7, 7
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      li a6, 7
// CHECK-NEXT:      scfgwi a7, 65
// CHECK-NEXT:      scfgwi a5, 97
// CHECK-NEXT:      scfgwi a6, 129
// CHECK-NEXT:      scfgwi a4, 193
// CHECK-NEXT:      li a4, -440
// CHECK-NEXT:      scfgwi a4, 225
// CHECK-NEXT:      li a4, -504
// CHECK-NEXT:      scfgwi a4, 257
// CHECK-NEXT:      li a4, 8
// CHECK-NEXT:      li a6, 7
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      scfgwi a6, 66
// CHECK-NEXT:      scfgwi a5, 98
// CHECK-NEXT:      scfgwi a4, 194
// CHECK-NEXT:      li a4, 8
// CHECK-NEXT:      scfgwi a4, 226
// CHECK-NEXT:      scfgwi t6, 832
// CHECK-NEXT:      scfgwi t5, 833
// CHECK-NEXT:      scfgwi t4, 802
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_0_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft4, 0(t4)
// CHECK-NEXT:      li t5, 7
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft4, ft0, ft1, ft4
// CHECK-NEXT:      fadd.d ft4, ft2, ft4
// CHECK-NEXT:      fmax.d ft4, ft4, ft3
// CHECK-NEXT:      fsd ft4, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_0_for
// CHECK-NEXT:  scf_body_end_0_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
