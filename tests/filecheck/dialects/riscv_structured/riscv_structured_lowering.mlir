// RUN: xdsl-opt -p lower-riscv-structured %s | filecheck %s

"builtin.module"() ({
  %file = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<s0>
  %success = "riscv_structured.syscall"(%file) {"syscall_num" = 64 : i32}: () -> !riscv.reg<s1>
// CHECK:      %file = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<s0>
// CHECK-NEXT: %{{.+}} = "riscv.mv"(%{{.+}}) : (!riscv.reg<s0>) -> !riscv.reg<a0>
// CHECK-NEXT: %{{.+}} = "riscv.li"() {"immediate" = 64 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT: "riscv.ecall"() : () -> ()
// CHECK-NEXT: %{{.+}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT: %{{.+}} = "riscv.mv"(%{{.+}}) : (!riscv.reg<a0>) -> !riscv.reg<s1>

  "riscv_structured.syscall"() {"syscall_num" = 93 : i32} : () -> ()
// CHECK-NEXT: %{{.+}} = "riscv.li"() {"immediate" = 93 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT: "riscv.ecall"() : () -> ()

}) : () -> ()
