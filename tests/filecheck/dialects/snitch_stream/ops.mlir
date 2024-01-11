// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%X, %Y, %Z, %n = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)

%pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<16>], "strides" = [#builtin.int<128>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
%X_str = "snitch_stream.strided_read"(%X, %pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
%Y_str = "snitch_stream.strided_read"(%Y, %pattern) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
%Z_str = "snitch_stream.strided_write"(%Z, %pattern) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<>>
"snitch_stream.generic"(%n, %X_str, %Y_str, %Z_str) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
^0(%x : !riscv.freg<>, %y : !riscv.freg<>):
    %z = riscv.fadd.d %x, %y : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    snitch_stream.yield %z : !riscv.freg<>
}) : (!riscv.reg<>, !stream.readable<!riscv.freg<>>, !stream.readable<!riscv.freg<>>, !stream.writable<!riscv.freg<>>) -> ()

// CHECK:       %X, %Y, %Z, %n = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:       %pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<16>], "strides" = [#builtin.int<128>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
// CHECK-NEXT:  %X_str = "snitch_stream.strided_read"(%X, %pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
// CHECK-NEXT:  %Y_str = "snitch_stream.strided_read"(%Y, %pattern) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
// CHECK-NEXT:  %Z_str = "snitch_stream.strided_write"(%Z, %pattern) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<>>
// CHECK-NEXT:  "snitch_stream.generic"(%n, %X_str, %Y_str, %Z_str) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
// CHECK-NEXT:  ^0(%x : !riscv.freg<>, %y : !riscv.freg<>):
// CHECK-NEXT:      %z = riscv.fadd.d %x, %y : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:      snitch_stream.yield %z : !riscv.freg<>
// CHECK-NEXT:  }) : (!riscv.reg<>, !stream.readable<!riscv.freg<>>, !stream.readable<!riscv.freg<>>, !stream.writable<!riscv.freg<>>) -> ()

// CHECK-GENERIC:       %X, %Y, %Z, %n = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-GENERIC-NEXT:       %pattern = "snitch_stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<16>], "strides" = [#builtin.int<128>, #builtin.int<8>], "dm" = #builtin.int<31>} : () -> !snitch_stream.stride_pattern_type<2>
// CHECK-GENERIC-NEXT:  %X_str = "snitch_stream.strided_read"(%X, %pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
// CHECK-GENERIC-NEXT:  %Y_str = "snitch_stream.strided_read"(%Y, %pattern) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
// CHECK-GENERIC-NEXT:  %Z_str = "snitch_stream.strided_write"(%Z, %pattern) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<>>
// CHECK-GENERIC-NEXT:  "snitch_stream.generic"(%n, %X_str, %Y_str, %Z_str) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
// CHECK-GENERIC-NEXT:  ^0(%x : !riscv.freg<>, %y : !riscv.freg<>):
// CHECK-GENERIC-NEXT:      %z = "riscv.fadd.d"(%x, %y) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:      "snitch_stream.yield"(%z) : (!riscv.freg<>) -> ()
// CHECK-GENERIC-NEXT:  }) : (!riscv.reg<>, !stream.readable<!riscv.freg<>>, !stream.readable<!riscv.freg<>>, !stream.writable<!riscv.freg<>>) -> ()
