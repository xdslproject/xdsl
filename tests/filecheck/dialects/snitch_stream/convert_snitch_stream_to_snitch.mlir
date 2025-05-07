// RUN: xdsl-opt -p convert-snitch-stream-to-snitch %s | filecheck %s

// CHECK: builtin.module {

%A, %B, %C = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
// CHECK-NEXT:  %A, %B, %C = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)

"snitch_stream.streaming_region"(%A, %B, %C) <{
    "stride_patterns" = [
        #snitch_stream.stride_pattern<ub = [2], strides = [8]>,
        #snitch_stream.stride_pattern<ub = [3, 2], strides = [8, 24]>,
        #snitch_stream.stride_pattern<ub = [5, 4, 3, 2], strides = [8, 40, 160, 480]>
    ],
    operandSegmentSizes = array<i32: 2, 1>
}> ({
^0(%a_stream : !snitch.readable<!riscv.freg<ft0>>, %b_stream : !snitch.readable<!riscv.freg<ft1>>, %c_stream : !snitch.writable<!riscv.freg<ft2>>):
    "test.op"() : () -> ()
}) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 2 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {dm = #builtin.int<0>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {dm = #builtin.int<0>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_stream_repetition"(%{{.*}}) {dm = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 2 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 3 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {dm = #builtin.int<1>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {dm = #builtin.int<1>, dimension = #builtin.int<1>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 24 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {dm = #builtin.int<1>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {dm = #builtin.int<1>, dimension = #builtin.int<1>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_stream_repetition"(%{{.*}}) {dm = #builtin.int<1>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 2 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 3 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 4 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 5 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {dm = #builtin.int<2>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {dm = #builtin.int<2>, dimension = #builtin.int<1>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {dm = #builtin.int<2>, dimension = #builtin.int<2>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {dm = #builtin.int<2>, dimension = #builtin.int<3>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 480 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 160 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 40 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {dm = #builtin.int<2>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {dm = #builtin.int<2>, dimension = #builtin.int<1>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {dm = #builtin.int<2>, dimension = #builtin.int<2>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {dm = #builtin.int<2>, dimension = #builtin.int<3>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_stream_repetition"(%{{.*}}) {dm = #builtin.int<2>} : (!riscv.reg) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%A) {dm = #builtin.int<0>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%B) {dm = #builtin.int<1>, dimension = #builtin.int<1>} : (!riscv.reg) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_destination"(%C) {dm = #builtin.int<2>, dimension = #builtin.int<3>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %a_stream, %b_stream, %c_stream = "snitch.ssr_enable"() : () -> (!snitch.readable<!riscv.freg<ft0>>, !snitch.readable<!riscv.freg<ft1>>, !snitch.writable<!riscv.freg<ft2>>)
// CHECK-NEXT:  "test.op"() : () -> ()
// CHECK-NEXT:  "snitch.ssr_disable"() : () -> ()

"snitch_stream.streaming_region"(%A, %B) <{
    "stride_patterns" = [
        #snitch_stream.stride_pattern<ub = [2], strides = [8]>
    ],
    operandSegmentSizes = array<i32: 1, 1>
}> ({
^0(%a_stream : !snitch.readable<!riscv.freg<ft0>>, %b_stream : !snitch.writable<!riscv.freg<ft1>>):
    "test.op"() : () -> ()
}) : (!riscv.reg, !riscv.reg) -> ()

// CHECK-NEXT:  %{{.*}} = riscv.li 2 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.addi %{{.*}}, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_bound"(%{{.*}}) {dm = #builtin.int<31>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 8 : !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_dimension_stride"(%{{.*}}) {dm = #builtin.int<31>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  "snitch.ssr_set_stream_repetition"(%{{.*}}) {dm = #builtin.int<31>} : (!riscv.reg) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_source"(%A) {dm = #builtin.int<0>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  "snitch.ssr_set_dimension_destination"(%B) {dm = #builtin.int<1>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-NEXT:  %{{.*}}, %{{.*}} = "snitch.ssr_enable"() : () -> (!snitch.readable<!riscv.freg<ft0>>, !snitch.writable<!riscv.freg<ft1>>)
// CHECK-NEXT:  "test.op"() : () -> ()
// CHECK-NEXT:  "snitch.ssr_disable"() : () -> ()

// CHECK-NEXT: }
