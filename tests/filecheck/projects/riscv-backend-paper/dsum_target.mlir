// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

  riscv.assembly_section ".text" {
    riscv.directive ".globl" "dsum"
    riscv.directive ".p2align" "2"
    riscv_func.func @dsum(%arg0 : !riscv.reg<a0>, %arg1 : !riscv.reg<a1>, %arg2 : !riscv.reg<a2>) -> !riscv.reg<a0> {
      %0 = riscv.mv %arg0 : (!riscv.reg<a0>) -> !riscv.reg<>
      %1 = riscv.mv %arg1 : (!riscv.reg<a1>) -> !riscv.reg<>
      %2 = riscv.mv %arg2 : (!riscv.reg<a2>) -> !riscv.reg<>
      %3 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<16>], "strides" = [#builtin.int<128>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
      %c0 = riscv.li 0 : () -> !riscv.reg<>
      %c1 = riscv.li 1 : () -> !riscv.reg<>
      %c128 = riscv.li 128 : () -> !riscv.reg<>
      "snitch_stream.streaming_region"(%0, %1, %2, %3) <{"operandSegmentSizes" = array<i32: 2, 1, 1>}> ({
      ^0(%5 : !stream.readable<!riscv.freg<ft0>>, %6 : !stream.readable<!riscv.freg<ft1>>, %7 : !stream.writable<!riscv.freg<ft2>>):
        riscv_scf.for %i : !riscv.reg<> = %c0 to %c128 step %c1 {
          %9 = riscv_snitch.read from %5 : !riscv.freg<ft0>
          %10 = riscv_snitch.read from %6 : !riscv.freg<ft1>
          %11 = riscv.fadd.d %9, %10 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
          riscv_snitch.write %11 to %7 : !riscv.freg<ft2>
          riscv_scf.yield
        }
      }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> ()
      %12 = riscv.mv %2 : (!riscv.reg<>) -> !riscv.reg<a0>
      riscv_func.return %12 : !riscv.reg<a0>
    }
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
