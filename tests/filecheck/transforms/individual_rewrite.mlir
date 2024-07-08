// RUN:xdsl-opt %s -p 'apply-individual-rewrite{matched_operation_index=4 operation_name="riscv.add" pattern_name="AddImmediates"}'| filecheck %s

%a = riscv.li 1 : !riscv.reg
%b = riscv.li 2 : !riscv.reg
%c = riscv.li 3 : !riscv.reg
%d = riscv.add %a, %b : (!riscv.reg, !riscv.reg) -> !riscv.reg
%e = riscv.add %b, %c : (!riscv.reg, !riscv.reg) -> !riscv.reg


//CHECK:         builtin.module {
// CHECK-NEXT:       %a = riscv.li 1 : !riscv.reg
// CHECK-NEXT:       %b = riscv.li 2 : !riscv.reg
// CHECK-NEXT:       %c = riscv.li 3 : !riscv.reg
// CHECK-NEXT:       %d = riscv.li 3 : !riscv.reg
// CHECK-NEXT:       %e = riscv.add %b, %c : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:     }
