// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

riscv.assembly_section ".text" {
  riscv.directive ".globl" "fill"
  riscv.directive ".p2align" "2"

  // y[ 16 x 16 ]
  riscv_func.func @fill(
    %X : !riscv.freg<fa0>,
    %Y : !riscv.reg<a0>
  ) {
    %X_moved = riscv.fmv.d %X : (!riscv.freg<fa0>) -> !riscv.freg<>
    %Y_moved = riscv.mv %Y : (!riscv.reg<a0>) -> !riscv.reg<>

    %x = riscv.fmv.d %X_moved : (!riscv.freg<>) -> !riscv.freg<>

    %stride_pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<256>], "strides" = [#builtin.int<8>], "dm" = #builtin.int<0>} : () -> !snitch_stream.stride_pattern_type<1>

    "snitch_stream.streaming_region"(%Y_moved, %stride_pattern) <{"operandSegmentSizes" = array<i32: 0, 1, 1>}> ({
    ^bb0(%Y_stream : !stream.writable<!riscv.freg<ft0>>):
      %c0 = riscv.li 0 : () -> !riscv.reg<>
      %c1 = riscv.li 1 : () -> !riscv.reg<>
      %c256 = riscv.li 256 : () -> !riscv.reg<>
      riscv_scf.for %i : !riscv.reg<> = %c0 to %c256 step %c1 {
        %y = riscv.fmv.d %x : (!riscv.freg<>) -> !riscv.freg<ft0>
        riscv_snitch.write %y to %Y_stream : !riscv.freg<ft0>
      }
    }) : (!riscv.reg<>, !snitch_stream.stride_pattern_type<1>) -> ()

    riscv_func.return
  }
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
