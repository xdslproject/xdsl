// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t riscv-asm %s | filecheck %s --check-prefix=CHECK-ASM


%0 = rv32.get_register : !riscv.reg<a0>
%1 = rv32.get_register : !riscv.reg<a1>
%2 = riscv.get_float_register : !riscv.freg<fa0>

riscv_debug.printf %0, %1, %2, "{}, {}, {}" : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.freg<fa0>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = rv32.get_register : !riscv.reg<a0>
// CHECK-NEXT:   %1 = rv32.get_register : !riscv.reg<a1>
// CHECK-NEXT:   %2 = riscv.get_float_register : !riscv.freg<fa0>
// CHECK-NEXT:   riscv_debug.printf %0, %1, %2 "{}, {}, {}" : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.freg<fa0>) -> ()
// CHECK-NEXT: }


// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %0 = "rv32.get_register"() : () -> !riscv.reg<a0>
// CHECK-GENERIC-NEXT:   %1 = "rv32.get_register"() : () -> !riscv.reg<a1>
// CHECK-GENERIC-NEXT:   %2 = "riscv.get_float_register"() : () -> !riscv.freg<fa0>
// CHECK-GENERIC-NEXT:   "riscv_debug.printf"(%0, %1, %2) {format_str = "{}, {}, {}"} : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.freg<fa0>) -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()

// CHECK-ASM: printf "{}, {}, {}", a0, a1, fa0
