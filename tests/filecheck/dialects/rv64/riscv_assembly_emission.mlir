// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

"builtin.module"() ({
  riscv_func.func @main() {
    %1 = riscv.li 5 : !riscv.reg<j_1>
    // CHECK: li j_1, 5
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

    // Terminate block
    riscv_func.return
    // CHECK-NEXT: ret
  }
}) : () -> ()
