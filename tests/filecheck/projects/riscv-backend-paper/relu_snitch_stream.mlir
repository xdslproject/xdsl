// RUN: xdsl-run %s | filecheck %s

builtin.module {
  riscv.assembly_section ".data" {
    riscv.label "a"
    riscv.directive ".word" "0x0,0x3ff00000,0x0,-0x40100000,0x0,0x0,0x0,0x40000000,0x0,-0x40000000,0x0,0x0"
  }
  riscv.assembly_section ".data" {
    riscv.label "b"
    riscv.directive ".word" "0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0"
  }
  riscv.assembly_section ".text" {
    riscv.directive ".globl" "main"
    riscv.directive ".p2align" "2"
    riscv_func.func @main() {
      %A = riscv.li "a" : () -> !riscv.reg<>
      %B = riscv.li "b" : () -> !riscv.reg<>
      %A_memref = builtin.unrealized_conversion_cast %A : !riscv.reg<> to memref<2x3xf64>
      printf.print_format "{}", %A_memref : memref<2x3xf64>
      %zero = riscv.get_register : () -> !riscv.reg<sp>
      %zero_1 = riscv.li 0 : () -> !riscv.reg<>
      riscv.sw %zero, %zero_1, -4 : (!riscv.reg<sp>, !riscv.reg<>) -> ()
      %zero_2 = riscv.li 0 : () -> !riscv.reg<>
      riscv.sw %zero, %zero_2, -8 : (!riscv.reg<sp>, !riscv.reg<>) -> ()
      %zero_3 = riscv.fld %zero, -8 : (!riscv.reg<sp>) -> !riscv.freg<>
      %stride_pattern = "snitch_stream.stride_pattern"() {"ub" = [#int<2>, #int<3>], "strides" = [#int<24>, #int<8>], "dm" = #int<31>} : () -> !snitch_stream.stride_pattern_type
      %a_stream = "snitch_stream.strided_read"(%A, %stride_pattern) {"dm" = #int<0>, "rank" = #int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.readable<!riscv.freg<ft0>>
      %b_stream = "snitch_stream.strided_write"(%B, %stride_pattern) {"dm" = #int<1>, "rank" = #int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.writable<!riscv.freg<ft1>>
      %c6 = riscv.li 6 : () -> !riscv.reg<>
      "snitch_stream.generic"(%c6, %a_stream, %b_stream) <{"operandSegmentSizes" = array<i32: 1, 1, 1>}> ({
      ^0(%a : !riscv.freg<ft0>):
        %res = riscv.fmax.d %a, %zero_3 : (!riscv.freg<ft0>, !riscv.freg<>) -> !riscv.freg<ft1>
        snitch_stream.yield %res : !riscv.freg<ft1>
      }) : (!riscv.reg<>, !stream.readable<!riscv.freg<ft0>>, !stream.writable<!riscv.freg<ft1>>) -> ()
      %B_memref = builtin.unrealized_conversion_cast %B : !riscv.reg<> to memref<2x3xf64>
      printf.print_format "{}", %B_memref : memref<2x3xf64>
      riscv_func.return
    }
  }
}

// CHECK: [[1.0, -1.0, 0.0], [2.0, -2.0, 0.0]]
// CHECK: [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
