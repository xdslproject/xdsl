// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

riscv_func.func @main() {
  // Assembler pseudo-instructions

  %li = rv64.li 1 : !riscv.reg
  // CHECK: %{{.*}} = rv64.li 1 : !riscv.reg

  // Load 64-bit value from memory
  %ld = rv64.ld %li, 8 : (!riscv.reg) -> !riscv.reg
  // CHECK: %{{.*}} = rv64.ld %{{.*}}, 8 : (!riscv.reg) -> !riscv.reg

  // Store 64-bit value to memory
  rv64.sd %li, %ld, 16 : (!riscv.reg, !riscv.reg) -> ()
  // CHECK: rv64.sd %{{.*}}, %{{.*}}, 16 : (!riscv.reg, !riscv.reg) -> ()

  riscv_func.return
}

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:      %li = "rv64.li"() {immediate = 1 : i64} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %ld = "rv64.ld"(%li) {immediate = 8 : i12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      "rv64.sd"(%li, %ld) {immediate = 16 : i12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()
