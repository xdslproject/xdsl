// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t riscv-asm %s | filecheck %s --check-prefix=CHECK-ASM

riscv_func.func @main() {
  %0 = rv64.li 5 : !riscv.reg<j_1>
  // CHECK: %{{.*}} = rv64.li 5 : !riscv.reg<j_1>
  %slli = rv64.slli %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT:      %{{.*}} = rv64.slli %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %srli = rv64.srli %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.srli %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %srai = rv64.srai %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.srai %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>

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
// CHECK-GENERIC-NEXT:      %0 = "rv64.li"() {immediate = 5 : i64} : () -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %slli = "rv64.slli"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %srli = "rv64.srli"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %srai = "rv64.srai"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %li = "rv64.li"() {immediate = 1 : i64} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %ld = "rv64.ld"(%li) {immediate = 8 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      "rv64.sd"(%li, %ld) {immediate = 16 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()

// CHECK-ASM:       .text
// CHECK-ASM-NEXT:  main:
// CHECK-ASM-NEXT:      li j_1, 5
// CHECK-ASM-NEXT:      slli j_1, j_1, 1
// CHECK-ASM-NEXT:      srli j_1, j_1, 1
// CHECK-ASM-NEXT:      srai j_1, j_1, 1
// CHECK-ASM-NEXT:      li , 1
// CHECK-ASM-NEXT:      ld , 8()
// CHECK-ASM-NEXT:      sd , 16()
// CHECK-ASM-NEXT:      ret
