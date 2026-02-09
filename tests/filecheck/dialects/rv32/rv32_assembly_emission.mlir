// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

riscv_func.func @main() {
  %0 = rv32.li 6 : !riscv.reg<zero>
  // CHECK:      li zero, 6
  %1 = rv32.li 5 : !riscv.reg<j_1>
  // CHECK-NEXT: li j_1, 5
  %slli = rv32.slli %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: slli j_1, j_1, 1
  %srli = rv32.srli %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: srli j_1, j_1, 1
  %srai = rv32.srai %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: srai j_1, j_1, 1

  %rori = rv32.rori %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: rori j_1, j_1, 1
  %bclri = rv32.bclri %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: bclri j_1, j_1, 1
  %bseti = rv32.bseti %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: bseti j_1, j_1, 1
  
  // Assembler pseudo-instructions
  %li = rv32.li 1 : !riscv.reg<j_0>
  // CHECK-NEXT: li j_0, 1
  
  // Terminate block
  riscv_func.return
  // CHECK-NEXT: ret
}
