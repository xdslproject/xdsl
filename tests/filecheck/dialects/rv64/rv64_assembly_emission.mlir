// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

riscv_func.func @main() {
  %0 = rv64.li 6 : !riscv.reg<zero>
  // CHECK:      li zero, 6
  %1 = rv64.li 5 : !riscv.reg<j_1>
  // CHECK-NEXT: li j_1, 5
  %slli = rv64.slli %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: slli j_1, j_1, 1
  %srli = rv64.srli %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: srli j_1, j_1, 1
  %srai = rv64.srai %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: srai j_1, j_1, 1

  %rori = rv64.rori %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: rori j_1, j_1, 1
  %roriw = rv64.roriw %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: roriw j_1, j_1, 1
  %bclri = rv64.bclri %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: bclri j_1, j_1, 1
  %bseti = rv64.bseti %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: bseti j_1, j_1, 1
  %slliuw = rv64.slli.uw %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: slli.uw j_1, j_1, 1
  
  // Assembler pseudo-instructions
  %li = rv64.li 1 : !riscv.reg<j_0>
  // CHECK-NEXT: li j_0, 1
  
  // Terminate block
  riscv_func.return
  // CHECK-NEXT: ret
}
