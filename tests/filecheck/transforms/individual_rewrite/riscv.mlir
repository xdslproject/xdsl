// RUN:xdsl-opt %s -p 'apply-individual-rewrite{matched_operation_index=4 operation_name="riscv.add" pattern_name="AddImmediates"}'| filecheck %s

%a = rv32.li 1 : !riscv.reg
%b = rv32.li 2 : !riscv.reg
%c = rv32.li 3 : !riscv.reg
%d = riscv.add %a, %b : (!riscv.reg, !riscv.reg) -> !riscv.reg
%e = riscv.add %b, %c : (!riscv.reg, !riscv.reg) -> !riscv.reg


//CHECK:         builtin.module {
// CHECK-NEXT:       %a = rv32.li 1 : !riscv.reg
// CHECK-NEXT:       %b = rv32.li 2 : !riscv.reg
// CHECK-NEXT:       %c = rv32.li 3 : !riscv.reg
// CHECK-NEXT:       %d = rv32.li 3 : !riscv.reg
// CHECK-NEXT:       %e = riscv.add %b, %c : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:     }
