// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t riscv-asm %s | filecheck %s --check-prefix=CHECK-ASM


%0 = rv32.get_register : !riscv.reg<a0>
%1 = rv32.get_register : !riscv.reg<a1>
%2 = riscv.get_float_register : !riscv.freg<fa0>

riscv_debug.printf "hello"
riscv_debug.printf "note" {comment = "no operands"}
riscv_debug.printf "value: {}" (%0 : !riscv.reg<a0>) {comment = "debug value"}
riscv_debug.printf "{}, {}, {}" (%0, %1, %2 : !riscv.reg<a0>, !riscv.reg<a1>, !riscv.freg<fa0>)

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = rv32.get_register : !riscv.reg<a0>
// CHECK-NEXT:   %1 = rv32.get_register : !riscv.reg<a1>
// CHECK-NEXT:   %2 = riscv.get_float_register : !riscv.freg<fa0>
// CHECK-NEXT:   riscv_debug.printf "hello"
// CHECK-NEXT:   riscv_debug.printf "note" {comment = "no operands"}
// CHECK-NEXT:   riscv_debug.printf "value: {}" (%0 : !riscv.reg<a0>) {comment = "debug value"}
// CHECK-NEXT:   riscv_debug.printf "{}, {}, {}" (%0, %1, %2 : !riscv.reg<a0>, !riscv.reg<a1>, !riscv.freg<fa0>)
// CHECK-NEXT: }


// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %0 = "rv32.get_register"() : () -> !riscv.reg<a0>
// CHECK-GENERIC-NEXT:   %1 = "rv32.get_register"() : () -> !riscv.reg<a1>
// CHECK-GENERIC-NEXT:   %2 = "riscv.get_float_register"() : () -> !riscv.freg<fa0>
// CHECK-GENERIC-NEXT:   "riscv_debug.printf"() {format_str = "hello"} : () -> ()
// CHECK-GENERIC-NEXT:   "riscv_debug.printf"() {format_str = "note", comment = "no operands"} : () -> ()
// CHECK-GENERIC-NEXT:   "riscv_debug.printf"(%0) {format_str = "value: {}", comment = "debug value"} : (!riscv.reg<a0>) -> ()
// CHECK-GENERIC-NEXT:   "riscv_debug.printf"(%0, %1, %2) {format_str = "{}, {}, {}"} : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.freg<fa0>) -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()

// CHECK-ASM: printf "hello"
// CHECK-ASM-NEXT: printf "note" {{ *}}# no operands
// CHECK-ASM-NEXT: printf "value: {}", a0 {{ *}}# debug value
// CHECK-ASM-NEXT: printf "{}, {}, {}", a0, a1, fa0
