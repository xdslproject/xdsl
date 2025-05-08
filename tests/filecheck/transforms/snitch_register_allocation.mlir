// RUN: xdsl-opt -p snitch-allocate-registers %s | filecheck %s

%ptr0, %ptr1, %ptr2 = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)

"snitch_stream.streaming_region"(%ptr0, %ptr1, %ptr2) <{
    "stride_patterns" = [#snitch_stream.stride_pattern<ub = [], strides = []>],
    operandSegmentSizes = array<i32: 2, 1>
}> ({
^0(%s0 : !snitch.readable<!riscv.freg>, %s1 : !snitch.readable<!riscv.freg>, %s2 : !snitch.writable<!riscv.freg>):
    %c5 = riscv.li 5 : !riscv.reg
    riscv_snitch.frep_outer %c5 {
        %x = riscv_snitch.read from %s0 : !riscv.freg
        %y = riscv_snitch.read from %s1 : !riscv.freg
        %r = riscv.fadd.d %x, %y : (!riscv.freg, !riscv.freg) -> !riscv.freg
        riscv_snitch.write %r to %s2 : !riscv.freg
    }
}) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()

// CHECK: builtin.module {

// CHECK-NEXT:    %ptr0, %ptr1, %ptr2 = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [], strides = []>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } ins(%ptr0, %ptr1 : !riscv.reg, !riscv.reg) outs(%ptr2 : !riscv.reg) {
// CHECK-NEXT:    ^{{.*}}(%s0 : !snitch.readable<!riscv.freg<ft0>>, %s1 : !snitch.readable<!riscv.freg<ft1>>, %s2 : !snitch.writable<!riscv.freg<ft2>>):
// CHECK-NEXT:      %c5 = riscv.li 5 : !riscv.reg
// CHECK-NEXT:      riscv_snitch.frep_outer %c5 {
// CHECK-NEXT:        %x = riscv_snitch.read from %s0 : !riscv.freg<ft0>
// CHECK-NEXT:        %y = riscv_snitch.read from %s1 : !riscv.freg<ft1>
// CHECK-NEXT:        %r = riscv.fadd.d %x, %y : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-NEXT:        riscv_snitch.write %r to %s2 : !riscv.freg<ft2>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// CHECK-NEXT: }
