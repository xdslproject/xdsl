// RUN: xdsl-opt -p riscv-allocate-registers -t riscv-asm %s | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<>
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<>
  %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
}) : () -> ()

// CHECK:      li j0, 6
// CHECK-NEXT: li j1, 5
// CHECK-NEXT: add j2, j0, j1
