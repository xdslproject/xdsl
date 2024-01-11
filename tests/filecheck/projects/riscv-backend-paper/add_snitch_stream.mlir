// RUN: xdsl-run %s | filecheck %s

builtin.module {
  riscv.assembly_section ".data" {
    riscv.label "a"
    riscv.directive ".word" "0x0,0x3ff00000,0x0,0x40000000,0x0,0x40080000,0x0,0x40100000,0x0,0x40140000,0x0,0x40180000"
    riscv.label "b"
    riscv.directive ".word" "0x0,0x0,0x0,0x3fd00000,0x0,0x3fe00000,0x0,0x3fe80000,0x0,0x3ff00000,0x0,0x3ff40000"
    riscv.label "c"
    riscv.directive ".word" "0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0"
  }
  riscv.assembly_section ".text" {
    riscv.directive ".globl" "main"
    riscv.directive ".p2align" "2"
    riscv_func.func @main() {
      %A = riscv.li "a" : () -> !riscv.reg<>
      %B = riscv.li "b" : () -> !riscv.reg<>
      %C = riscv.li "c" : () -> !riscv.reg<>
      %0 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>, #builtin.int<3>], "strides" = [#builtin.int<24>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
      %1 = "snitch_stream.strided_read"(%A, %0) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<ft0>>
      %2 = "snitch_stream.strided_read"(%B, %0) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<ft1>>
      %3 = "snitch_stream.strided_write"(%C, %0) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<ft2>>
      %4 = riscv.li 6 : () -> !riscv.reg<>
      "snitch_stream.generic"(%4, %1, %2, %3) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
      ^0(%a : !riscv.freg<ft0>, %b : !riscv.freg<ft1>):
        %sum = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
        snitch_stream.yield %sum : !riscv.freg<ft2>
      }) : (!riscv.reg<>, !stream.readable<!riscv.freg<ft0>>, !stream.readable<!riscv.freg<ft1>>, !stream.writable<!riscv.freg<ft2>>) -> ()
      %5 = riscv.mv %C : (!riscv.reg<>) -> !riscv.reg<>
      %v0 = riscv.fld %5, 0 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v0_1 = builtin.unrealized_conversion_cast %v0 : !riscv.freg<> to f64
      %v1 = riscv.fld %C, 8 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v1_1 = builtin.unrealized_conversion_cast %v1 : !riscv.freg<> to f64
      %v2 = riscv.fld %C, 16 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v2_1 = builtin.unrealized_conversion_cast %v2 : !riscv.freg<> to f64
      %v3 = riscv.fld %C, 24 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v3_1 = builtin.unrealized_conversion_cast %v3 : !riscv.freg<> to f64
      %v4 = riscv.fld %C, 32 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v4_1 = builtin.unrealized_conversion_cast %v4 : !riscv.freg<> to f64
      %v5 = riscv.fld %C, 40 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v5_1 = builtin.unrealized_conversion_cast %v5 : !riscv.freg<> to f64
      printf.print_format "[[{}, {}, {}], [{}, {}, {}]]", %v0_1 : f64, %v1_1 : f64, %v2_1 : f64, %v3_1 : f64, %v4_1 : f64, %v5_1 : f64
      riscv_func.return
    }
  }
}

// CHECK: [[1.0, 2.25, 3.5], [4.75, 6.0, 7.25]]
