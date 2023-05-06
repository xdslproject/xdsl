// RUN: xdsl-opt -p lower-riscv-ssa %s | filecheck %s

"builtin.module"() ({
  %success = "riscv_ssa.syscall"() {"syscall_num" = 64 : i32}: () -> (!riscv.reg<s1>)
// CHECK:      %{{.+}} = "riscv.li"() {"immediate" = 64 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT: "riscv.ecall"() : () -> ()
// CHECK-NEXT: %{{.+}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT: %{{.+}} = "riscv.mv"(%{{.+}}) : (!riscv.reg<a0>) -> !riscv.reg<s1>

  "riscv_ssa.syscall"() {"syscall_num" = 93 : i32} : () -> ()
// CHECK-NEXT: %{{.+}} = "riscv.li"() {"immediate" = 93 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT: "riscv.ecall"() : () -> ()

}) : () -> ()
