// RUN: xdsl-opt -p snitch-allocate-registers %s | filecheck %s

%stride_pattern, %ptr0, %ptr1, %ptr2 = "test.op"() : () -> (!snitch_stream.stride_pattern_type<2>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
%s0 = "snitch_stream.strided_read"(%ptr0, %stride_pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
%s1 = "snitch_stream.strided_read"(%ptr1, %stride_pattern) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<>>
%s2 = "snitch_stream.strided_write"(%ptr2, %stride_pattern) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<>>
%c128 = riscv.li 128 : () -> !riscv.reg<>

"snitch_stream.generic"(%c128, %s0, %s1, %s2) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
^0(%x : !riscv.freg<>, %y : !riscv.freg<>):
    %r0 = riscv.fadd.d %x, %y : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    snitch_stream.yield %r0 : !riscv.freg<>
}) : (!riscv.reg<>, !stream.readable<!riscv.freg<>>, !stream.readable<!riscv.freg<>>, !stream.writable<!riscv.freg<>>) -> ()

"snitch_stream.generic"(%c128, %s1, %s0, %s2) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
^0(%x : !riscv.freg<>, %y : !riscv.freg<>):
    %r0 = riscv.fadd.d %x, %y : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    snitch_stream.yield %r0 : !riscv.freg<>
}) : (!riscv.reg<>, !stream.readable<!riscv.freg<>>, !stream.readable<!riscv.freg<>>, !stream.writable<!riscv.freg<>>) -> ()

// CHECK: builtin.module {

// CHECK-NEXT:  %stride_pattern, %ptr0, %ptr1, %ptr2 = "test.op"() : () -> (!snitch_stream.stride_pattern_type<2>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:  %s0 = "snitch_stream.strided_read"(%ptr0, %stride_pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<ft0>>
// CHECK-NEXT:  %s1 = "snitch_stream.strided_read"(%ptr1, %stride_pattern) {"dm" = #builtin.int<1>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.readable<!riscv.freg<ft1>>
// CHECK-NEXT:  %s2 = "snitch_stream.strided_write"(%ptr2, %stride_pattern) {"dm" = #builtin.int<2>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type<2>) -> !stream.writable<!riscv.freg<ft2>>
// CHECK-NEXT:  %c128 = riscv.li 128 : () -> !riscv.reg<>
// CHECK-NEXT:  "snitch_stream.generic"(%c128, %s0, %s1, %s2) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
// CHECK-NEXT:  ^0(%{{.*}} : !riscv.freg<ft0>, %{{.*}} : !riscv.freg<ft1>):
// CHECK-NEXT:    %{{.*}} = riscv.fadd.d %{{.*}}, %{{.*}} : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-NEXT:    snitch_stream.yield %{{.*}} : !riscv.freg<ft2>
// CHECK-NEXT:  }) : (!riscv.reg<>, !stream.readable<!riscv.freg<ft0>>, !stream.readable<!riscv.freg<ft1>>, !stream.writable<!riscv.freg<ft2>>) -> ()
// CHECK-NEXT:  "snitch_stream.generic"(%c128, %s1, %s0, %s2) <{"operandSegmentSizes" = array<i32: 1, 2, 1>}> ({
// CHECK-NEXT:  ^1(%{{.*}} : !riscv.freg<ft1>, %{{.*}} : !riscv.freg<ft0>):
// CHECK-NEXT:    %{{.*}} = riscv.fadd.d %{{.*}}, %{{.*}} : (!riscv.freg<ft1>, !riscv.freg<ft0>) -> !riscv.freg<ft2>
// CHECK-NEXT:    snitch_stream.yield %{{.*}} : !riscv.freg<ft2>
// CHECK-NEXT:  }) : (!riscv.reg<>, !stream.readable<!riscv.freg<ft1>>, !stream.readable<!riscv.freg<ft0>>, !stream.writable<!riscv.freg<ft2>>) -> ()

// CHECK-NEXT: }
