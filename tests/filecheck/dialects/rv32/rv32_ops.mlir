// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t riscv-asm %s | filecheck %s --check-prefix=CHECK-ASM

riscv_func.func @main() {
  %0 = rv32.li 5 : !riscv.reg<j_1>
  // CHECK: %{{.*}} = rv32.li 5 : !riscv.reg<j_1>
  %slli = rv32.slli %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT:      %{{.*}} = rv32.slli %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %srli = rv32.srli %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv32.srli %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %srai = rv32.srai %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv32.srai %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %bclri = rv32.bclri %0, 1: (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv32.bclri %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %bexti = rv32.bexti %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK:      %{{.*}} = rv32.bexti %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %binvi = rv32.binvi %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv32.binvi %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %bseti = rv32.bseti %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv32.bseti %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  %rori = rv32.rori %0, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
  // CHECK-NEXT: %{{.*}} = rv32.rori %{{.*}}, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>

  riscv_func.return
}


// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:      %0 = "rv32.li"() {immediate = 5 : i32} : () -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %slli = "rv32.slli"(%0) {immediate = 1 : ui5} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %srli = "rv32.srli"(%0) {immediate = 1 : ui5} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %srai = "rv32.srai"(%0) {immediate = 1 : ui5} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %bclri = "rv32.bclri"(%0) {immediate = 1 : ui5} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %bexti = "rv32.bexti"(%0) {immediate = 1 : ui5} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %binvi = "rv32.binvi"(%0) {immediate = 1 : ui5} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %bseti = "rv32.bseti"(%0) {immediate = 1 : ui5} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
// CHECK-GENERIC-NEXT:      %rori = "rv32.rori"(%0) {immediate = 1 : ui5} : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
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
// CHECK-ASM-NEXT:      ret
