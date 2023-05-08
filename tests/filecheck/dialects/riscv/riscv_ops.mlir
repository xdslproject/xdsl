// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %0 = "riscv.get_register"() : () -> !riscv.reg<>
  %1 = "riscv.get_register"() : () -> !riscv.reg<>
  // RV32I/RV64I: 2.4 Integer Computational Instructions

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
  %mv = "riscv.mv"(%0) : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK: %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<>) -> !riscv.reg<>

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

  // RV32I/RV64I: 2.5 Control Transfer Instructions

  // Unconditional Branch Instructions
  %jal_i = "riscv.jal"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %jal_i = "riscv.jal"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
  %jal_r = "riscv.jal"() {"immediate" = #riscv.label<"label">}: () -> !riscv.reg<>
  // CHECK-NEXT: %jal_r = "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> !riscv.reg<>

  %j_i = "riscv.j"() {"immediate" = 1 : i32}: () -> !riscv.reg<zero>
  // CHECK-NEXT: %j_i = "riscv.j"() {"immediate" = 1 : i32} : () -> !riscv.reg<zero>
  %j_r = "riscv.j"() {"immediate" = #riscv.label<"label">}: () -> !riscv.reg<zero>
  // CHECK-NEXT: %j_r = "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> !riscv.reg<zero>

  %jalr_i = "riscv.jalr"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %jalr_i = "riscv.jalr"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %jalr_r = "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %jalr_r = "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<>) -> !riscv.reg<>

  "riscv.ret"() : () -> ()
  // CHECK-NEXT: "riscv.ret"() : () -> ()


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

  // RV32I/RV64I: 2.6 Load and Store Instructions

  %lb = "riscv.lb"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lb"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %lbu = "riscv.lbu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lbu"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %lh = "riscv.lh"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lh"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %lhu = "riscv.lhu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lhu"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %lw = "riscv.lw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lw"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  "riscv.sb"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.sb"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.sh"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.sh"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.sw"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: "riscv.sw"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()

  // RV32I/RV64I: 2.8 Control and Status Register Instructions

  %csrrw_rw = "riscv.csrrw"(%0) {"csr" = 1024 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrw"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %csrrw_w = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"} : (!riscv.reg<>) -> !riscv.reg<>
  %csrrs_rw = "riscv.csrrs"(%0) {"csr" = 1024 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrs"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<>) -> !riscv.reg<>
  %csrrc_rw = "riscv.csrrc"(%0) {"csr" = 1024 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrc"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
  %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<>) -> !riscv.reg<>
  %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<>
  %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<>
  %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<>
  %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<>
  %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32} : () -> !riscv.reg<>
  %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32} : () -> !riscv.reg<>

  // Machine Mode Privileged Instructions
  "riscv.wfi"() : () -> ()
  // CHECK-NEXT: "riscv.wfi"() : () -> ()


  // RV32M/RV64M: 7 “M” Standard Extension for Integer Multiplication and Division

  // Multiplication Operations
  %mul = "riscv.mul"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.mul"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %mulh = "riscv.mulh"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.mulh"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %mulhsu = "riscv.mulhsu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.mulhsu"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %mulhu = "riscv.mulhu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.mulhu"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

  // Division Operations
  %div = "riscv.div"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.div"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %divu = "riscv.divu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.divu"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %rem = "riscv.rem"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.rem"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %remu = "riscv.remu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.remu"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

  // Assembler pseudo-instructions

  %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
  // Environment Call and Breakpoints
  "riscv.ecall"() : () -> ()
  // CHECK-NEXT: "riscv.ecall"() : () -> ()
  "riscv.ebreak"() : () -> ()
  // CHECK-NEXT: "riscv.ebreak"() : () -> ()

}) : () -> ()
