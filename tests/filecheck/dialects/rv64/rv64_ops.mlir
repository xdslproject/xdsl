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
  %bclri = rv64.bclri %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.bclri %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %bexti = rv64.bexti %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.bexti %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %binvi = rv64.binvi %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.binvi %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %bseti = rv64.bseti %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.bseti %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %rori = rv64.rori %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.rori %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %srliw = rv64.srliw %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.srliw %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %roriw = rv64.roriw %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv64.roriw %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>

  riscv_func.return
}

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:      %0 = "rv64.li"() {immediate = 5 : i64} : () -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %slli = "rv64.slli"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %srli = "rv64.srli"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %srai = "rv64.srai"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %bclri = "rv64.bclri"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %bexti = "rv64.bexti"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %binvi = "rv64.binvi"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %bseti = "rv64.bseti"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %rori = "rv64.rori"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %srliw = "rv64.srliw"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %roriw = "rv64.roriw"(%0) {immediate = 1 : ui6} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()

// CHECK-ASM:       .text
// CHECK-ASM-NEXT:  main:
// CHECK-ASM-NEXT:      li j_1, 5
// CHECK-ASM-NEXT:      slli j_1, j_1, 1
// CHECK-ASM-NEXT:      srli j_1, j_1, 1
// CHECK-ASM-NEXT:      srai j_1, j_1, 1
// CHECK-ASM-NEXT:      bclri j_1, j_1, 1
// CHECK-ASM-NEXT:      bexti j_1, j_1, 1
// CHECK-ASM-NEXT:      binvi j_1, j_1, 1
// CHECK-ASM-NEXT:      bseti j_1, j_1, 1
// CHECK-ASM-NEXT:      rori j_1, j_1, 1
// CHECK-ASM-NEXT:      srliw j_1, j_1, 1
// CHECK-ASM-NEXT:      roriw j_1, j_1, 1
// CHECK-ASM-NEXT:      ret
