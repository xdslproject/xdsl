// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

riscv_func.func @main() {
  // Assembler pseudo-instructions

  %li = rv64.li 1 : !riscv.reg
  // CHECK: %{{.*}} = rv64.li 1 : !riscv.reg

  riscv_func.return
}

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:      %li = "rv64.li"() {immediate = 1 : i64} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()
