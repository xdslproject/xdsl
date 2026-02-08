// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

riscv_func.func @main() {
  %0 = rv64.li 6 : !riscv.reg<zero>
  // CHECK:      li zero, 6
  %1 = rv64.li 5 : !riscv.reg<j_1>
  // CHECK-NEXT: li j_1, 5

  // Assembler pseudo-instructions
  %li = rv64.li 1 : !riscv.reg<j_0>
  // CHECK-NEXT: li j_0, 1
  
  // Terminate block
  riscv_func.return
  // CHECK-NEXT: ret
}
