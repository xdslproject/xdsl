// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

riscv_func.func @main() {
  %0 = riscv.get_register : () -> !riscv.reg<>
  %1 = riscv.get_register : () -> !riscv.reg<>

  // RISC-V extensions
  %scfgw = riscv_snitch.scfgw %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  // CHECK: %scfgw = riscv_snitch.scfgw %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  %scfgwi_zero = riscv_snitch.scfgwi %0, 42 : (!riscv.reg<>) -> !riscv.reg<zero>
  // CHECK-NEXT: %scfgwi_zero = riscv_snitch.scfgwi %0, 42 : (!riscv.reg<>) -> !riscv.reg<zero>

  riscv_snitch.frep_outer %0 {
    %add_o = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  }
  // CHECK-NEXT:  riscv_snitch.frep_outer %0 {
  // CHECK-NEXT:    %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT:  }

  riscv_snitch.frep_inner %0 {
    %add_i = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  }
  // CHECK-NEXT:  riscv_snitch.frep_inner %0 {
  // CHECK-NEXT:    %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT:  }

  // Terminate block
  riscv_func.return
}

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:     %0 = "riscv.get_register"() : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %1 = "riscv.get_register"() : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %scfgw = "riscv_snitch.scfgw"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
// CHECK-GENERIC-NEXT:     %scfgwi_zero = "riscv_snitch.scfgwi"(%0) {"immediate" = 42 : si12} : (!riscv.reg<>) -> !riscv.reg<zero>
// CHECK-GENERIC-NEXT:    "riscv_snitch.frep_outer"(%{{.*}}) ({
// CHECK-GENERIC-NEXT:      %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:      "riscv_snitch.frep_yield"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:    "riscv_snitch.frep_inner"(%{{.*}}) ({
// CHECK-GENERIC-NEXT:      %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:      "riscv_snitch.frep_yield"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) {"sym_name" = "main", "function_type" = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
