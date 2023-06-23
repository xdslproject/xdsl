// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s
"builtin.module"() ({
  %0 = "riscv.get_integer_register"() : () -> !riscv.ireg<>
  %1 = "riscv.get_integer_register"() : () -> !riscv.ireg<>
  // RV32I/RV64I: 2.4 Integer Computational Instructions

  // Integer Register-Immediate Instructions
  %addi = "riscv.addi"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK: %{{.*}} = "riscv.addi"(%{{.*}}) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %slti = "riscv.slti"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.slti"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %sltiu = "riscv.sltiu"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sltiu"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %andi = "riscv.andi"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.andi"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %ori = "riscv.ori"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.ori"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %xori = "riscv.xori"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.xori"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %slli = "riscv.slli"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.slli"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %srli = "riscv.srli"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.srli"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %srai = "riscv.srai"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.srai"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %lui = "riscv.lui"() {"immediate" = 1 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lui"() {"immediate" = 1 : i32} : () -> !riscv.ireg<>
  %auipc = "riscv.auipc"() {"immediate" = 1 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.auipc"() {"immediate" = 1 : i32} : () -> !riscv.ireg<>
  %mv = "riscv.mv"(%0) : (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK: %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.ireg<>

  // Integer Register-Register Operations
  %add = "riscv.add"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %slt = "riscv.slt"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.slt"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %sltu = "riscv.sltu"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %and = "riscv.and"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.and"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %or = "riscv.or"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.or"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %xor = "riscv.xor"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.xor"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %sll = "riscv.sll"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sll"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %srl = "riscv.srl"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.srl"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %sub = "riscv.sub"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sub"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %sra = "riscv.sra"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.sra"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  "riscv.nop"() : () -> ()
  // CHECK-NEXT: "riscv.nop"() : () -> ()

  // RV32I/RV64I: 2.5 Control Transfer Instructions
  // terminators continue at the end of module

  // Unconditional Branch Instructions
  "riscv.jal"() {"immediate" = 1 : i32} : () -> ()
  // CHECK-NEXT: "riscv.jal"() {"immediate" = 1 : i32} : () -> ()
  "riscv.jal"() {"immediate" = 1 : i32, "rd" = !riscv.ireg<>} : () -> ()
  // CHECK-NEXT: "riscv.jal"() {"immediate" = 1 : i32, "rd" = !riscv.ireg<>} : () -> ()
  "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()

  "riscv.j"() {"immediate" = 1 : i32} : () -> ()
  // CHECK-NEXT: "riscv.j"() {"immediate" = 1 : i32} : () -> ()
  "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> ()

  "riscv.jalr"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.jalr"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> ()
  "riscv.jalr"(%0) {"immediate" = 1 : i32, "rd" = !riscv.ireg<>} : (!riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.jalr"(%0) {"immediate" = 1 : i32, "rd" = !riscv.ireg<>} : (!riscv.ireg<>) -> ()
  "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.ireg<>) -> ()

  // Conditional Branch Instructions
  "riscv.beq"(%0, %1) {"offset" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.beq"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "riscv.bne"(%0, %1) {"offset" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.bne"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "riscv.blt"(%0, %1) {"offset" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.blt"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "riscv.bge"(%0, %1) {"offset" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.bge"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "riscv.bltu"(%0, %1) {"offset" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.bltu"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "riscv.bgeu"(%0, %1) {"offset" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.bgeu"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()

  // RV32I/RV64I: 2.6 Load and Store Instructions

  %lb = "riscv.lb"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lb"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %lbu = "riscv.lbu"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lbu"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %lh = "riscv.lh"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lh"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %lhu = "riscv.lhu"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lhu"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %lw = "riscv.lw"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.lw"(%0) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  "riscv.sb"(%0, %1) {"immediate" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.sb"(%0, %1) {"immediate" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "riscv.sh"(%0, %1) {"immediate" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.sh"(%0, %1) {"immediate" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "riscv.sw"(%0, %1) {"immediate" = 1 : i32}: (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.sw"(%0, %1) {"immediate" = 1 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()

  // RV32I/RV64I: 2.8 Control and Status Register Instructions

  %csrrw_rw = "riscv.csrrw"(%0) {"csr" = 1024 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrw"(%0) {"csr" = 1024 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %csrrw_w = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"} : (!riscv.ireg<>) -> !riscv.ireg<>
  %csrrs_rw = "riscv.csrrs"(%0) {"csr" = 1024 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrs"(%0) {"csr" = 1024 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.ireg<>) -> !riscv.ireg<>
  %csrrc_rw = "riscv.csrrc"(%0) {"csr" = 1024 : i32}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrc"(%0) {"csr" = 1024 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.ireg<>) -> !riscv.ireg<>
  %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.ireg<>
  %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.ireg<>
  %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.ireg<>
  %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.ireg<>
  %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32} : () -> !riscv.ireg<>
  %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32} : () -> !riscv.ireg<>

  // Machine Mode Privileged Instructions
  "riscv.wfi"() : () -> ()
  // CHECK-NEXT: "riscv.wfi"() : () -> ()


  // RV32M/RV64M: 7 “M” Standard Extension for Integer Multiplication and Division

  // Multiplication Operations
  %mul = "riscv.mul"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.mul"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %mulh = "riscv.mulh"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.mulh"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %mulhsu = "riscv.mulhsu"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.mulhsu"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %mulhu = "riscv.mulhu"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.mulhu"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>

  // Division Operations
  %div = "riscv.div"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.div"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %divu = "riscv.divu"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.divu"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %rem = "riscv.rem"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.rem"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  %remu = "riscv.remu"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.remu"(%{{.*}}, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>

  // Assembler pseudo-instructions

  %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.ireg<>
  // Environment Call and Breakpoints
  "riscv.ecall"() : () -> ()
  // CHECK-NEXT: "riscv.ecall"() : () -> ()
  "riscv.ebreak"() : () -> ()
  // CHECK-NEXT: "riscv.ebreak"() : () -> ()
  "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
  // CHECK-NEXT: "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
  "riscv.directive"() ({
    %nested_li = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.ireg<>
  }) {"directive" = ".text"} : () -> ()
  // CHECK-NEXT:  "riscv.directive"() ({
  // CHECK-NEXT:    %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.ireg<>
  // CHECK-NEXT:  }) {"directive" = ".text"} : () -> ()

  // Custom instruction
  %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.ireg<>, !riscv.ireg<>) -> (!riscv.ireg<>, !riscv.ireg<>)
  // CHECK-NEXT:   %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.ireg<>, !riscv.ireg<>) -> (!riscv.ireg<>, !riscv.ireg<>)


  // RISC-V extensions
  "riscv.scfgw"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK-NEXT: "riscv.scfgw"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> ()

  // RV32I/RV64I: 2.5 Control Transfer Instructions (cont'd)
  // terminators

  // RV32F: 8 “F” Standard Extension forSingle-Precision Floating-Point, Version 2.0
  %f0 = "riscv.get_float_register"() : () -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.get_float_register"() : () -> !riscv.freg<>
  %f1 = "riscv.get_float_register"() : () -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.get_float_register"() : () -> !riscv.freg<>
  %f2 = "riscv.get_float_register"() : () -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.get_float_register"() : () -> !riscv.freg<>

  %fmadd_s = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fmadd.s"(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fmsub_s = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fmsub.s"(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fnmsub_s = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fnmsub.s"(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fnmadd_s = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fnmadd.s"(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

  %fadd_s = "riscv.fadd.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fadd.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fsub_s = "riscv.fsub.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fsub.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fmul_s = "riscv.fmul.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fmul.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fdiv_s = "riscv.fdiv.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fdiv.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fsqrt_s = "riscv.fsqrt.s"(%f0) : (!riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fsqrt.s"(%{{.*}}) : (!riscv.freg<>) -> !riscv.freg<>

  %fsgnj_s = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fsgnj.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fsgnjn_s = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fsgnjn.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fsgnjx_s = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fsgnjx.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

  %fmin_s = "riscv.fmin.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fmin.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fmax_s = "riscv.fmax.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fmax.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

  %fcvt_w_s = "riscv.fcvt.w.s"(%f0) : (!riscv.freg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fcvt.w.s"(%{{.*}}) : (!riscv.freg<>) -> !riscv.ireg<>
  %fcvt_wu_s = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fcvt.wu.s"(%{{.*}}) : (!riscv.freg<>) -> !riscv.ireg<>
  %fmv_x_w = "riscv.fmv.x.w"(%f0) : (!riscv.freg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fmv.x.w"(%{{.*}}) : (!riscv.freg<>) -> !riscv.ireg<>

  %feq_s = "riscv.feq.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.ireg<>
  %flt_s = "riscv.flt.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.ireg<>
  %fle_s = "riscv.fle.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.ireg<>
  %fclass_s = "riscv.fclass.s"(%f0) : (!riscv.freg<>) -> !riscv.ireg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fclass.s"(%{{.*}}) : (!riscv.freg<>) -> !riscv.ireg<>
  %fcvt_s_w = "riscv.fcvt.s.w"(%0) : (!riscv.ireg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.w"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.freg<>
  %fcvt_s_wu = "riscv.fcvt.s.wu"(%0) : (!riscv.ireg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.wu"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.freg<>
  %fmv_w_x = "riscv.fmv.w.x"(%0) : (!riscv.ireg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.fmv.w.x"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.freg<>

  %flw = "riscv.flw"(%0) {"immediate" = 1 : i32}: (!riscv.ireg<>) -> !riscv.freg<>
  // CHECK-NEXT: %{{.*}} = "riscv.flw"(%{{.*}}) {"immediate" = 1 : i32} : (!riscv.ireg<>) -> !riscv.freg<>
  "riscv.fsw"(%0, %f0) {"immediate" = 1 : i32} : (!riscv.ireg<>, !riscv.freg<>) -> ()
  // CHECK-NEXT: "riscv.fsw"(%{{.*}}, %{{.*}}) {"immediate" = 1 : i32} : (!riscv.ireg<>, !riscv.freg<>) -> ()

  // Unconditional Branch Instructions
  "riscv.ret"() : () -> ()
  // CHECK-NEXT: "riscv.ret"() : () -> ()
}) : () -> ()
