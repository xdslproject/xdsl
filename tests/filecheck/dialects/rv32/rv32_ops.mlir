// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

riscv_func.func @main() {
  %0 = riscv.get_register : !riscv.reg

  %slli = rv32.slli %0, 1: (!riscv.reg) -> !riscv.reg
  // CHECK: %{{.*}} = rv32.slli %0, 1 : (!riscv.reg) -> !riscv.reg
  %srli = rv32.srli %0, 1: (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{.*}} = rv32.srli %0, 1 : (!riscv.reg) -> !riscv.reg
  %srai = rv32.srai %0, 1: (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{.*}} = rv32.srai %0, 1 : (!riscv.reg) -> !riscv.reg
  %rori = rv32.rori %0, 1 : (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{.*}} = rv32.rori %{{.*}}, 1 : (!riscv.reg) -> !riscv.reg
  
  // Assembler pseudo-instructions
  %li = rv32.li 1 : !riscv.reg
  // CHECK: %{{.*}} = rv32.li 1 : !riscv.reg


  riscv_func.return
}

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:      %0 = "riscv.get_register"() : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %slli = "rv32.slli"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srli = "rv32.srli"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srai = "rv32.srai"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %rori = "rv32.rori"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %li = "rv32.li"() {immediate = 1 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()
