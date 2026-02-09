// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%X, %Y, %Z = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)

snitch_stream.streaming_region {
    stride_patterns = [
        #snitch_stream.stride_pattern<ub = [8, 16], strides = [128, 8]>
    ]
} ins(%X, %Y : !riscv.reg, !riscv.reg) outs(%Z : !riscv.reg) {
^bb0(%a_stream : !snitch.readable<!riscv.freg<ft0>>, %b_stream : !snitch.readable<!riscv.freg<ft1>>, %c_stream : !snitch.writable<!riscv.freg<ft2>>):
    %c5 = rv32.li 5 : !riscv.reg
    riscv_snitch.frep_outer %c5 {
        %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
        %b = riscv_snitch.read from %b_stream : !riscv.freg<ft1>
        %c = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
        riscv_snitch.write %c to %c_stream : !riscv.freg<ft2>
    }
}

"test.op"() {with_repeat = #snitch_stream.stride_pattern<ub = [8, 16], strides = [128, 8], repeat = 8>} : () -> ()

// CHECK:       %X, %Y, %Z = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [8, 16], strides = [128, 8]>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } ins(%X, %Y : !riscv.reg, !riscv.reg) outs(%Z : !riscv.reg) {
// CHECK-NEXT:    ^bb0(%a_stream : !snitch.readable<!riscv.freg<ft0>>, %b_stream : !snitch.readable<!riscv.freg<ft1>>, %c_stream : !snitch.writable<!riscv.freg<ft2>>):
// CHECK-NEXT:      %c5 = rv32.li 5 : !riscv.reg
// CHECK-NEXT:      riscv_snitch.frep_outer %c5 {
// CHECK-NEXT:        %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
// CHECK-NEXT:        %b = riscv_snitch.read from %b_stream : !riscv.freg<ft1>
// CHECK-NEXT:        %c = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-NEXT:        riscv_snitch.write %c to %c_stream : !riscv.freg<ft2>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// CHECK-GENERIC:         %X, %Y, %Z = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
// CHECK-GENERIC-NEXT:    "snitch_stream.streaming_region"(%X, %Y, %Z) <{stride_patterns = [#snitch_stream.stride_pattern<ub = [8, 16], strides = [128, 8]>], operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^bb0(%a_stream : !snitch.readable<!riscv.freg<ft0>>, %b_stream : !snitch.readable<!riscv.freg<ft1>>, %c_stream : !snitch.writable<!riscv.freg<ft2>>):
// CHECK-GENERIC-NEXT:      %c5 = "rv32.li"() {immediate = 5 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv_snitch.frep_outer"(%c5) ({
// CHECK-GENERIC-NEXT:        %a = "riscv_snitch.read"(%a_stream) : (!snitch.readable<!riscv.freg<ft0>>) -> !riscv.freg<ft0>
// CHECK-GENERIC-NEXT:        %b = "riscv_snitch.read"(%b_stream) : (!snitch.readable<!riscv.freg<ft1>>) -> !riscv.freg<ft1>
// CHECK-GENERIC-NEXT:        %c = "riscv.fadd.d"(%a, %b) {fastmath = #riscv.fastmath<none>} : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-GENERIC-NEXT:        "riscv_snitch.write"(%c, %c_stream) : (!riscv.freg<ft2>, !snitch.writable<!riscv.freg<ft2>>) -> ()
// CHECK-GENERIC-NEXT:        "riscv_snitch.frep_yield"() : () -> ()
// CHECK-GENERIC-NEXT:      }) {stagger_mask = 0 : i4, stagger_count = 0 : i3} : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:    }) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
