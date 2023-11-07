// RUN: xdsl-opt -p convert-snitch-stream-to-snitch %s | filecheck %s

// CHECK: builtin.module {

%A, %B, %C = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:  %A, %B, %C = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)

%0 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>, #builtin.int<3>], "strides" = [#builtin.int<24>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type
// CHECK-NEXT:  %0 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:  %1 = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:  %2 = riscv.li 24 : () -> !riscv.reg<>
// CHECK-NEXT:  %3 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:  %4 = riscv.addi %0, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %5 = riscv.addi %1, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%4) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%5) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%2) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %6 = riscv.mul %4, %2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %7 = riscv.sub %3, %6 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%7) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()


%1 = "snitch_stream.strided_read"(%A, %0) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.readable<!riscv.freg<ft0>>
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%A) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %a = riscv.get_float_register : () -> !riscv.freg<ft0>

%2 = "snitch_stream.strided_read"(%B, %0) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.readable<!riscv.freg<ft1>>
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%B) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %b = riscv.get_float_register : () -> !riscv.freg<ft1>

%3 = "snitch_stream.strided_write"(%C, %0) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.writable<!riscv.freg<ft2>>
// CHECK-NEXT:  "snitch.ssr_set_dimension_destination"(%C) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()


%4 = riscv.li 6 : () -> !riscv.reg<>
// CHECK-NEXT:  %8 = riscv.li 6 : () -> !riscv.reg<>


"snitch_stream.generic"(%4, %1, %2, %3) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
^0(%a : !riscv.freg<ft0>, %b : !riscv.freg<ft1>):
%sum = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
snitch_stream.yield %sum : !riscv.freg<ft2>
}) : (!riscv.reg<>, !stream.readable<!riscv.freg<ft0>>, !stream.readable<!riscv.freg<ft1>>, !stream.writable<!riscv.freg<ft2>>) -> ()
// CHECK-NEXT:  "snitch.ssr_enable"() : () -> ()
// CHECK-NEXT:  %9 = riscv.addi %8, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  riscv_snitch.frep_outer %9, 0, 0 ({
// CHECK-NEXT:    %sum = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-NEXT:    riscv_snitch.frep_yield %sum : (!riscv.freg<ft2>) -> ()
// CHECK-NEXT:  }) : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_disable"() : () -> ()

// CHECK-NEXT: }

