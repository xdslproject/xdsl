// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

riscv_func.func @main() {
  // Assembler pseudo-instructions

  %li = rv64.li 1 : !riscv.reg
  // CHECK: %{{.*}} = rv64.li 1 : !riscv.reg
  riscv.assembly_section ".text" attributes {"foo" = i32} {
    %nested_li = rv64.li 1 : !riscv.reg
  }
  // CHECK-NEXT:  riscv.assembly_section ".text" attributes {foo = i32} {
  // CHECK-NEXT:    %{{.*}} = rv64.li 1 : !riscv.reg
  // CHECK-NEXT:  }

  riscv.assembly_section ".text" {
    %nested_li = rv64.li 1 : !riscv.reg
  }
  // CHECK-NEXT:  riscv.assembly_section ".text" {
  // CHECK-NEXT:    %{{.*}} = rv64.li 1 : !riscv.reg
  // CHECK-NEXT:  }

  riscv_func.return
}

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:      %li = "rv64.li"() {immediate = 1 : i64} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv.assembly_section"() ({
// CHECK-GENERIC-NEXT:        %nested_li = "rv64.li"() {immediate = 1 : i64} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      }) {directive = ".text", foo = i32} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.assembly_section"() ({
// CHECK-GENERIC-NEXT:        %nested_li = "rv64.li"() {immediate = 1 : i64} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      }) {directive = ".text"} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()
