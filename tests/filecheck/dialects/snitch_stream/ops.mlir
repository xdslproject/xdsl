// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%X, %Y, %Z, %n = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)

%pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<16>], "strides" = [#builtin.int<128>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
%X_str = "snitch_stream.strided_read"(%X, %pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
%Y_str = "snitch_stream.strided_read"(%Y, %pattern) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
%Z_str = "snitch_stream.strided_write"(%Z, %pattern) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<>>

"snitch_stream.streaming_region"(%X, %Y, %Z, %pattern) <{"operandSegmentSizes" = array<i32: 2, 1, 1>}> ({
^0(%a_stream : !stream.readable<!riscv.freg<ft0>>, %b_stream : !stream.readable<!riscv.freg<ft1>>, %c_stream : !stream.writable<!riscv.freg<ft2>>):
    %c5 = riscv.li 5 : () -> !riscv.reg<>
    riscv_snitch.frep_outer %c5 {
        %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
        %b = riscv_snitch.read from %b_stream : !riscv.freg<ft1>
        %c = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
        riscv_snitch.write %c to %c_stream : !riscv.freg<ft2>
    }
}) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> ()


// CHECK:       %X, %Y, %Z, %n = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:       %pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<16>], "strides" = [#builtin.int<128>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
// CHECK-NEXT:  %X_str = "snitch_stream.strided_read"(%X, %pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
// CHECK-NEXT:  %Y_str = "snitch_stream.strided_read"(%Y, %pattern) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
// CHECK-NEXT:  %Z_str = "snitch_stream.strided_write"(%Z, %pattern) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<>>
// CHECK-NEXT:  "snitch_stream.streaming_region"(%X, %Y, %Z, %pattern) <{"operandSegmentSizes" = array<i32: 2, 1, 1>}> ({
// CHECK-NEXT:  ^0(%a_stream : !stream.readable<!riscv.freg<ft0>>, %b_stream : !stream.readable<!riscv.freg<ft1>>, %c_stream : !stream.writable<!riscv.freg<ft2>>):
// CHECK-NEXT:    %c5 = riscv.li 5 : () -> !riscv.reg<>
// CHECK-NEXT:    riscv_snitch.frep_outer %c5 {
// CHECK-NEXT:      %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
// CHECK-NEXT:      %b = riscv_snitch.read from %b_stream : !riscv.freg<ft1>
// CHECK-NEXT:      %c = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-NEXT:      riscv_snitch.write %c to %c_stream : !riscv.freg<ft2>
// CHECK-NEXT:    }
// CHECK-NEXT:  }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> ()


// CHECK-GENERIC:       %X, %Y, %Z, %n = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-GENERIC-NEXT:       %pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<16>], "strides" = [#builtin.int<128>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
// CHECK-GENERIC-NEXT:  %X_str = "snitch_stream.strided_read"(%X, %pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
// CHECK-GENERIC-NEXT:  %Y_str = "snitch_stream.strided_read"(%Y, %pattern) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
// CHECK-GENERIC-NEXT:  %Z_str = "snitch_stream.strided_write"(%Z, %pattern) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<>>
// CHECK-GENERIC-NEXT:    "snitch_stream.streaming_region"(%X, %Y, %Z, %pattern) <{"operandSegmentSizes" = array<i32: 2, 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^0(%a_stream : !stream.readable<!riscv.freg<ft0>>, %b_stream : !stream.readable<!riscv.freg<ft1>>, %c_stream : !stream.writable<!riscv.freg<ft2>>):
// CHECK-GENERIC-NEXT:      %c5 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:      "riscv_snitch.frep_outer"(%c5) ({
// CHECK-GENERIC-NEXT:        %a = "riscv_snitch.read"(%a_stream) : (!stream.readable<!riscv.freg<ft0>>) -> !riscv.freg<ft0>
// CHECK-GENERIC-NEXT:        %b = "riscv_snitch.read"(%b_stream) : (!stream.readable<!riscv.freg<ft1>>) -> !riscv.freg<ft1>
// CHECK-GENERIC-NEXT:        %c = "riscv.fadd.d"(%a, %b) {"fastmath" = #riscv.fastmath<none>} : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-GENERIC-NEXT:        "riscv_snitch.write"(%c, %c_stream) : (!riscv.freg<ft2>, !stream.writable<!riscv.freg<ft2>>) -> ()
// CHECK-GENERIC-NEXT:        "riscv_snitch.frep_yield"() : () -> ()
// CHECK-GENERIC-NEXT:      }) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> ()
