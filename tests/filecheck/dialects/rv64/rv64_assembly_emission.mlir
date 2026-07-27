// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

riscv_func.func @main() {
  %0 = rv64.li 6 : !riscv.reg<zero>
  // CHECK:      li zero, 6
  %1 = rv64.li 5 : !riscv.reg<j_1>
  // CHECK-NEXT: li j_1, 5

  // Assembler pseudo-instructions
  %li = rv64.li 1 : !riscv.reg<j_0>
  // CHECK-NEXT: li j_0, 1

  // Load 64-bit value: ld rd, offset(rs1)
  %ld = rv64.ld %1, 8 : (!riscv.reg<j_1>) -> !riscv.reg<j_2>
  // CHECK-NEXT: ld j_2, 8(j_1)

  // Store 64-bit value: sd rs2, offset(rs1)
  rv64.sd %1, %ld, 16 : (!riscv.reg<j_1>, !riscv.reg<j_2>) -> ()
  // CHECK-NEXT: sd j_2, 16(j_1)

  // Terminate block
  riscv_func.return
  // CHECK-NEXT: ret
}
