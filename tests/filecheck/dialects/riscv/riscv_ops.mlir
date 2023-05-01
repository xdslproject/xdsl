// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %0 = "test.op"() : () -> !riscv.reg<>
  %1 = "test.op"() : () -> !riscv.reg<>
  // RV32I/RV64I: Integer Computational Instructions (Section 2.4)
  // Integer Register-Immediate Instructions
  %addi = "riscv.addi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK: %{{.*}} = "riscv.addi"(%{{.*}}) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %slti = "riscv.slti"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.slti"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %sltiu = "riscv.sltiu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sltiu"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %andi = "riscv.andi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.andi"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %ori = "riscv.ori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.ori"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %xori = "riscv.xori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.xori"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %slli = "riscv.slli"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.slli"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %srli = "riscv.srli"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.srli"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %srai = "riscv.srai"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.srai"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %lui = "riscv.lui"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lui"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
  %auipc = "riscv.auipc"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.auipc"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
  // Integer Register-Register Operations
  %add = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %slt = "riscv.slt"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.slt"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %sltu = "riscv.sltu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %and = "riscv.and"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.and"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %or = "riscv.or"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.or"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %xor = "riscv.xor"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.xor"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
   %sll = "riscv.sll"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sll"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %srl = "riscv.srl"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.srl"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %sub = "riscv.sub"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sub"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %sra = "riscv.sra"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sra"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  "riscv.nop"() : () -> ()
  // CHECK-NEXT: "riscv.nop"() : () -> ()
  // Conditional Branch Instructions
  "riscv.beq"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.beq"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.bne"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.bne"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.blt"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.blt"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.bge"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.bge"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.bltu"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.bltu"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.bgeu"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.bgeu"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // Assembler pseudo-insgtructions
  %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
}) : () -> ()
