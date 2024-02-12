// RUN: xdsl-opt -p convert-snitch-stream-to-snitch %s | filecheck %s

// CHECK: builtin.module {

%A, %B, %C = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:  %A, %B, %C = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)

%sp1 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>], "strides" = [#builtin.int<8>], "dm" = #builtin.int<0>} : () -> !snitch_stream.stride_pattern_type<1>
%sp2 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>, #builtin.int<3>], "strides" = [#builtin.int<24>, #builtin.int<8>], "dm" = #builtin.int<1>} : () -> !snitch_stream.stride_pattern_type<2>
%sp4 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>, #builtin.int<3>, #builtin.int<4>, #builtin.int<5>], "strides" = [#builtin.int<480>, #builtin.int<160>, #builtin.int<40>, #builtin.int<8>], "dm" = #builtin.int<2>} : () -> !snitch_stream.stride_pattern_type<4>


"snitch_stream.streaming_region"(%A, %B, %C, %sp1, %sp2, %sp4) <{"operandSegmentSizes" = array<i32: 2, 1, 3>}> ({
^0(%a_stream : !stream.readable<!riscv.freg<ft0>>, %b_stream : !stream.readable<!riscv.freg<ft1>>, %c_stream : !stream.writable<!riscv.freg<ft2>>):
    "test.op"() : () -> ()
}) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<1>, !snitch_stream.stride_pattern_type<2>, !snitch_stream.stride_pattern_type<4>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 24 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 5 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 480 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 160 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 40 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<2>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<2>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%A) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%B) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_destination"(%C) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %a_stream, %b_stream, %c_stream = "snitch.ssr_enable"() : () -> (!stream.readable<!riscv.freg<ft0>>, !stream.readable<!riscv.freg<ft1>>, !stream.writable<!riscv.freg<ft2>>)
// CHECK-NEXT:  "test.op"() : () -> ()
// CHECK-NEXT:  "snitch.ssr_disable"() : () -> ()


%sp1_31 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>], "strides" = [#builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<1>

"snitch_stream.streaming_region"(%A, %B, %sp1_31) <{"operandSegmentSizes" = array<i32: 1, 1, 1>}> ({
^0(%a_stream : !stream.readable<!riscv.freg<ft0>>, %b_stream : !stream.writable<!riscv.freg<ft1>>):
    "test.op"() : () -> ()
}) : (!riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<1>) -> ()

// CHECK-NEXT:  %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%A) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_destination"(%B) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}}, %{{.*}} = "snitch.ssr_enable"() : () -> (!stream.readable<!riscv.freg<ft0>>, !stream.writable<!riscv.freg<ft1>>)
// CHECK-NEXT:  "test.op"() : () -> ()
// CHECK-NEXT:  "snitch.ssr_disable"() : () -> ()

// CHECK-NEXT: }

