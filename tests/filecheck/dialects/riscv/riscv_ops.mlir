// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %0 = "test.op"() : () -> !riscv.reg<>
  %1 = "test.op"() : () -> !riscv.reg<>
  %add = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK: %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %sub = "riscv.sub"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sub"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
  %xor = "riscv.xor"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.xor"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %addi = "riscv.addi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.addi"(%{{.*}}) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %slti = "riscv.slti"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.slti"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %sltiu = "riscv.sltiu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sltiu"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %xori = "riscv.xori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.xori"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %ori = "riscv.ori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.ori"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %andi = "riscv.andi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.andi"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
}) : () -> ()
