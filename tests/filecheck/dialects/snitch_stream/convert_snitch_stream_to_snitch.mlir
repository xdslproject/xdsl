// RUN: xdsl-opt -p convert-snitch-stream-to-snitch %s | filecheck %s

// CHECK: builtin.module {

%A, %B, %C = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:  %A, %B, %C = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)

%sp1 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>], "strides" = [#builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<1>
// CHECK-NEXT:  %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : () -> !riscv.reg<>

%sp2 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>, #builtin.int<3>], "strides" = [#builtin.int<24>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
// CHECK-NEXT:  %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 24 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()


%sp4 = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<2>, #builtin.int<3>, #builtin.int<4>, #builtin.int<5>], "strides" = [#builtin.int<480>, #builtin.int<160>, #builtin.int<40>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<4>
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
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<2>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<2>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {"dm" = #builtin.int<31>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()

%1 = "snitch_stream.strided_read"(%A, %sp2) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<ft0>>
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%A) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv_snitch.get_stream : !stream.readable<!riscv.freg<ft0>>

%2 = "snitch_stream.strided_read"(%B, %sp2) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<ft1>>
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%B) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv_snitch.get_stream : !stream.readable<!riscv.freg<ft1>>

%3 = "snitch_stream.strided_write"(%C, %sp2) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<ft2>>
// CHECK-NEXT:  "snitch.ssr_set_dimension_destination"(%C) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %{{.*}} = riscv_snitch.get_stream : !stream.writable<!riscv.freg<ft2>>


"snitch_stream.streaming_region"(%A, %B, %C, %sp2) <{"operandSegmentSizes" = array<i32: 2, 1, 1>}> ({
^0(%a_stream : !stream.readable<!riscv.freg<ft0>>, %b_stream : !stream.readable<!riscv.freg<ft1>>, %c_stream : !stream.writable<!riscv.freg<ft2>>):
    %c5 = riscv.li 5 : () -> !riscv.reg<>
    riscv_snitch.frep_outer %c5 {
        %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
        %b = riscv_snitch.read from %b_stream : !riscv.freg<ft1>
        %c = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
        riscv_snitch.write %c to %c_stream : !riscv.freg<ft2>
    }
}) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%A) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %a_stream = riscv_snitch.get_stream : !stream.readable<!riscv.freg<ft0>>
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%B) {"dm" = #builtin.int<1>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %b_stream = riscv_snitch.get_stream : !stream.readable<!riscv.freg<ft1>>
// CHECK-NEXT:  "snitch.ssr_set_dimension_destination"(%C) {"dm" = #builtin.int<2>, "dimension" = #builtin.int<1>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:  %c_stream = riscv_snitch.get_stream : !stream.writable<!riscv.freg<ft2>>
// CHECK-NEXT:  "snitch.ssr_enable"() : () -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 5 : () -> !riscv.reg<>
// CHECK-NEXT:  riscv_snitch.frep_outer %{{.*}} {
// CHECK-NEXT:    %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
// CHECK-NEXT:    %b = riscv_snitch.read from %b_stream : !riscv.freg<ft1>
// CHECK-NEXT:    %c = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-NEXT:    riscv_snitch.write %c to %c_stream : !riscv.freg<ft2>
// CHECK-NEXT:  }
// CHECK-NEXT:  "snitch.ssr_disable"() : () -> ()

// CHECK-NEXT: }

