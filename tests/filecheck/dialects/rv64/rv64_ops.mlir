// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

riscv_func.func @main() {
  %0 = riscv.get_register : !riscv.reg

  %slli = rv64.slli %0, 1: (!riscv.reg) -> !riscv.reg
  // CHECK: %{{.*}} = rv64.slli %0, 1 : (!riscv.reg) -> !riscv.reg
  %srli = rv64.srli %0, 1: (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{.*}} = rv64.srli %0, 1 : (!riscv.reg) -> !riscv.reg
  %srai = rv64.srai %0, 1: (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{.*}} = rv64.srai %0, 1 : (!riscv.reg) -> !riscv.reg
  %srliw = rv64.srliw %0, 1: (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{.*}} = rv64.srliw %0, 1 : (!riscv.reg) -> !riscv.reg
  %rori = rv64.rori %0, 1 : (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{.*}} = rv64.rori %{{.*}}, 1 : (!riscv.reg) -> !riscv.reg
  %roriw = rv64.roriw %0, 1 : (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{.*}} = rv64.roriw %{{.*}}, 1 : (!riscv.reg) -> !riscv.reg
  
  // Assembler pseudo-instructions
  %li = rv64.li 1 : !riscv.reg
  // CHECK: %{{.*}} = rv64.li 1 : !riscv.reg

  riscv_func.return
}

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:      %0 = "riscv.get_register"() : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %slli = "rv64.slli"(%0) {immediate = 1 : ui6} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srli = "rv64.srli"(%0) {immediate = 1 : ui6} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srai = "rv64.srai"(%0) {immediate = 1 : ui6} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srliw = "rv64.srliw"(%0) {immediate = 1 : ui6} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %rori = "rv64.rori"(%0) {immediate = 1 : ui6} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %roriw = "rv64.roriw"(%0) {immediate = 1 : ui6} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %li = "rv64.li"() {immediate = 1 : i64} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()
