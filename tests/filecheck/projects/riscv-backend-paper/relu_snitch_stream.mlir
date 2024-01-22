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
      %zero = riscv.get_register : () -> !riscv.reg<sp>
      %zero_1 = riscv.li 0 : () -> !riscv.reg<>
      riscv.sw %zero, %zero_1, -4 : (!riscv.reg<sp>, !riscv.reg<>) -> ()
      %zero_2 = riscv.li 0 : () -> !riscv.reg<>
      riscv.sw %zero, %zero_2, -8 : (!riscv.reg<sp>, !riscv.reg<>) -> ()
      %zero_3 = riscv.fld %zero, -8 : (!riscv.reg<sp>) -> !riscv.freg<>
      %stride_pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>, #builtin.int<3>], "strides" = [#builtin.int<24>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
      "snitch_stream.streaming_region"(%A, %B, %stride_pattern, %stride_pattern) <{"operandSegmentSizes" = array<i32: 1, 1, 2>}> ({
      ^0(%a_stream : !stream.readable<!riscv.freg<ft0>>, %b_stream : !stream.writable<!riscv.freg<ft1>>):
        %c5 = riscv.li 5 : () -> !riscv.reg<>
        riscv_snitch.frep_outer %c5 {
          %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
          %b = riscv.fmax.d %a, %zero_3 : (!riscv.freg<ft0>, !riscv.freg<>) -> !riscv.freg<ft1>
          riscv_snitch.write %b to %b_stream : !riscv.freg<ft1>
        }
      }) : (!riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<2>, !snitch_stream.stride_pattern_type<2>) -> ()
      %v0 = riscv.fld %B, 0 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v0_1 = builtin.unrealized_conversion_cast %v0 : !riscv.freg<> to f64
      %v1 = riscv.fld %B, 8 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v1_1 = builtin.unrealized_conversion_cast %v1 : !riscv.freg<> to f64
      %v2 = riscv.fld %B, 16 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v2_1 = builtin.unrealized_conversion_cast %v2 : !riscv.freg<> to f64
      %v3 = riscv.fld %B, 24 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v3_1 = builtin.unrealized_conversion_cast %v3 : !riscv.freg<> to f64
      %v4 = riscv.fld %B, 32 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v4_1 = builtin.unrealized_conversion_cast %v4 : !riscv.freg<> to f64
      %v5 = riscv.fld %B, 40 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg<>) -> !riscv.freg<>
      %v5_1 = builtin.unrealized_conversion_cast %v5 : !riscv.freg<> to f64
      printf.print_format "[[{}, {}, {}], [{}, {}, {}]]", %v0_1 : f64, %v1_1 : f64, %v2_1 : f64, %v3_1 : f64, %v4_1 : f64, %v5_1 : f64
      riscv_func.return
    }
  }
}

// CHECK: [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
