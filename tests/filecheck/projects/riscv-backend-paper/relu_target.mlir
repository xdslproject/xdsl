// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

riscv.assembly_section ".text" {
  riscv.directive ".globl" "relu"
  riscv.directive ".p2align" "2"
  riscv_func.func @relu(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>) {
    %X_moved = riscv.mv %X : (!riscv.reg<a0>) -> !riscv.reg<>
    %Y_moved = riscv.mv %Y : (!riscv.reg<a1>) -> !riscv.reg<>

    %zero_int = riscv.get_register : () -> !riscv.reg<zero>
    %zero_float = riscv.fcvt.d.w %zero_int : (!riscv.reg<zero>) -> !riscv.freg<>

    %stride_pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<16>, #builtin.int<16>], "strides" = [#builtin.int<128>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>

    "snitch_stream.streaming_region"(%X_moved, %Y_moved, %stride_pattern) <{"operandSegmentSizes" = array<i32: 1, 1, 1>}> ({
    ^0(%X_stream : !stream.readable<!riscv.freg<ft0>>, %Y_stream : !stream.writable<!riscv.freg<ft1>>):
      %c0 = riscv.li 0 : () -> !riscv.reg<>
      %c1 = riscv.li 1 : () -> !riscv.reg<>
      %c256 = riscv.li 256 : () -> !riscv.reg<>
      riscv_scf.for %i : !riscv.reg<> = %c0 to %c256 step %c1 {
        %x = riscv_snitch.read from %X_stream : !riscv.freg<ft0>
        %y = riscv.fmax.d %x, %zero_float : (!riscv.freg<ft0>, !riscv.freg<>) -> !riscv.freg<ft1>
        riscv_snitch.write %y to %Y_stream : !riscv.freg<ft1>
      }
    }) : (!riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> ()

    riscv_func.return
  }
}

// CHECK:       .text
// CHECK-NEXT:  .globl relu
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  relu:
// CHECK-NEXT:      mv t1, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t2, 128
// CHECK-NEXT:      li t4, 15
// CHECK-NEXT:      li t3, 15
// CHECK-NEXT:      scfgwi t4, 95
// CHECK-NEXT:      scfgwi t3, 127
// CHECK-NEXT:      scfgwi t2, 223
// CHECK-NEXT:      li t2, -1912
// CHECK-NEXT:      scfgwi t2, 255
// CHECK-NEXT:      scfgwi t1, 800
// CHECK-NEXT:      scfgwi t0, 929
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t0, 255
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fmax.d ft1, ft0, ft3
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
