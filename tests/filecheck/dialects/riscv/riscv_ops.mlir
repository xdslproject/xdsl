// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s
"builtin.module"() ({
  %0 = "riscv.get_register"() : () -> !riscv.reg<x$>
  %1 = "riscv.get_register"() : () -> !riscv.reg<x$>
  // RV32I/RV64I: 2.4 Integer Computational Instructions

  // Integer Register-Immediate Instructions
  %addi = "riscv.addi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK: %{{.*}} = "riscv.addi"(%{{.*}}) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %slti = "riscv.slti"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.slti"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %sltiu = "riscv.sltiu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.sltiu"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %andi = "riscv.andi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.andi"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %ori = "riscv.ori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.ori"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %xori = "riscv.xori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.xori"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %slli = "riscv.slli"(%0) {"immediate" = 1 : ui5}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.slli"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %srli = "riscv.srli"(%0) {"immediate" = 1 : ui5}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.srli"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %srai = "riscv.srai"(%0) {"immediate" = 1 : ui5}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.srai"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lui = "riscv.lui"() {"immediate" = 1 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.lui"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
  %auipc = "riscv.auipc"() {"immediate" = 1 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.auipc"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
  %mv = "riscv.mv"(%0) : (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK: %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.reg<x$>

  // Integer Register-Register Operations
  %add = "riscv.add"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %slt = "riscv.slt"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.slt"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %sltu = "riscv.sltu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %and = "riscv.and"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.and"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %or = "riscv.or"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.or"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %xor = "riscv.xor"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.xor"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %sll = "riscv.sll"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.sll"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %srl = "riscv.srl"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.srl"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %sub = "riscv.sub"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.sub"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %sra = "riscv.sra"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.sra"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  "riscv.nop"() : () -> ()
  // CHECK-NEXT: "riscv.nop"() : () -> ()

  // RV32I/RV64I: 2.5 Control Transfer Instructions
  // terminators continue at the end of module

  // Unconditional Branch Instructions
  "riscv.jal"() {"immediate" = 1 : i32} : () -> ()
  // CHECK-NEXT: "riscv.jal"() {"immediate" = 1 : i32} : () -> ()
  "riscv.jal"() {"immediate" = 1 : i32, "rd" = !riscv.reg<x$>} : () -> ()
  // CHECK-NEXT: "riscv.jal"() {"immediate" = 1 : i32, "rd" = !riscv.reg<x$>} : () -> ()
  "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()

  "riscv.j"() {"immediate" = 1 : i32} : () -> ()
  // CHECK-NEXT: "riscv.j"() {"immediate" = 1 : i32} : () -> ()
  "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> ()

  "riscv.jalr"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.jalr"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> ()
  "riscv.jalr"(%0) {"immediate" = 1 : i32, "rd" = !riscv.reg<x$>} : (!riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.jalr"(%0) {"immediate" = 1 : i32, "rd" = !riscv.reg<x$>} : (!riscv.reg<x$>) -> ()
  "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<x$>) -> ()

  // Conditional Branch Instructions
  "riscv.beq"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.beq"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.bne"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.bne"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.blt"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.blt"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.bge"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.bge"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.bltu"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.bltu"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.bgeu"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.bgeu"(%{{.*}}, %{{.*}}) {"offset" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()

  // RV32I/RV64I: 2.6 Load and Store Instructions

  %lb = "riscv.lb"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.lb"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lbu = "riscv.lbu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.lbu"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lh = "riscv.lh"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.lh"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lhu = "riscv.lhu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.lhu"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lw = "riscv.lw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.lw"(%0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  "riscv.sb"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.sb"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.sh"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.sh"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.sw"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.sw"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()

  // RV32I/RV64I: 2.8 Control and Status Register Instructions

  %csrrw_rw = "riscv.csrrw"(%0) {"csr" = 1024 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrw"(%0) {"csr" = 1024 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrw_w = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrs_rw = "riscv.csrrs"(%0) {"csr" = 1024 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrs"(%0) {"csr" = 1024 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrc_rw = "riscv.csrrc"(%0) {"csr" = 1024 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrc"(%0) {"csr" = 1024 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<x$>
  %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<x$>
  %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<x$>
  %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<x$>
  %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32} : () -> !riscv.reg<x$>
  %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32} : () -> !riscv.reg<x$>

  // Machine Mode Privileged Instructions
  "riscv.wfi"() : () -> ()
  // CHECK-NEXT: "riscv.wfi"() : () -> ()


  // RV32M/RV64M: 7 “M” Standard Extension for Integer Multiplication and Division

  // Multiplication Operations
  %mul = "riscv.mul"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.mul"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %mulh = "riscv.mulh"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.mulh"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %mulhsu = "riscv.mulhsu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.mulhsu"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %mulhu = "riscv.mulhu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.mulhu"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

  // Division Operations
  %div = "riscv.div"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.div"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %divu = "riscv.divu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.divu"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %rem = "riscv.rem"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.rem"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %remu = "riscv.remu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.remu"(%{{.*}}, %{{.*}}) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

  // Assembler pseudo-instructions

  %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
  // Environment Call and Breakpoints
  "riscv.ecall"() : () -> ()
  // CHECK-NEXT: "riscv.ecall"() : () -> ()
  "riscv.ebreak"() : () -> ()
  // CHECK-NEXT: "riscv.ebreak"() : () -> ()
  "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
  // CHECK-NEXT: "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
  "riscv.directive"() ({
    %nested_li = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
  }) {"directive" = ".text"} : () -> ()
  // CHECK-NEXT:  "riscv.directive"() ({
  // CHECK-NEXT:    %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
  // CHECK-NEXT:  }) {"directive" = ".text"} : () -> ()

  // Custom instruction
  %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.reg<x$>, !riscv.reg<x$>) -> (!riscv.reg<x$>, !riscv.reg<x$>)
  // CHECK-NEXT:   %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.reg<x$>, !riscv.reg<x$>) -> (!riscv.reg<x$>, !riscv.reg<x$>)


  // RISC-V extensions
  "riscv.scfgw"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: "riscv.scfgw"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()

  // RV32I/RV64I: 2.5 Control Transfer Instructions (cont'd)
  // terminators

  // RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0
  %f0 = "riscv.get_float_register"() : () -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.get_float_register"() : () -> !riscv.freg<f$>
  %f1 = "riscv.get_float_register"() : () -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.get_float_register"() : () -> !riscv.freg<f$>
  %f2 = "riscv.get_float_register"() : () -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.get_float_register"() : () -> !riscv.freg<f$>

  %fmadd_s = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fmadd.s"(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fmsub_s = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fmsub.s"(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fnmsub_s = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fnmsub.s"(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fnmadd_s = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fnmadd.s"(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>

  %fadd_s = "riscv.fadd.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fadd.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fsub_s = "riscv.fsub.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fsub.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fmul_s = "riscv.fmul.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fmul.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fdiv_s = "riscv.fdiv.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fdiv.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fsqrt_s = "riscv.fsqrt.s"(%f0) : (!riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fsqrt.s"(%{{.*}}) : (!riscv.freg<f$>) -> !riscv.freg<f$>

  %fsgnj_s = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fsgnj.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fsgnjn_s = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fsgnjn.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fsgnjx_s = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fsgnjx.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>

  %fmin_s = "riscv.fmin.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fmin.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fmax_s = "riscv.fmax.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fmax.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>

  %fcvt_w_s = "riscv.fcvt.w.s"(%f0) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.fcvt.w.s"(%{{.*}}) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  %fcvt_wu_s = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.fcvt.wu.s"(%{{.*}}) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  %fmv_x_w = "riscv.fmv.x.w"(%f0) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.fmv.x.w"(%{{.*}}) : (!riscv.freg<f$>) -> !riscv.reg<x$>

  %feq_s = "riscv.feq.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  %flt_s = "riscv.flt.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  %fle_s = "riscv.fle.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  %fclass_s = "riscv.fclass.s"(%f0) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %{{.*}} = "riscv.fclass.s"(%{{.*}}) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  %fcvt_s_w = "riscv.fcvt.s.w"(%0) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.w"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  %fcvt_s_wu = "riscv.fcvt.s.wu"(%0) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.wu"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  %fmv_w_x = "riscv.fmv.w.x"(%0) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.fmv.w.x"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.freg<f$>

  %flw = "riscv.flw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %{{.*}} = "riscv.flw"(%{{.*}}) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.freg<f$>
  "riscv.fsw"(%0, %f0) {"immediate" = 1 : i32} : (!riscv.reg<x$>, !riscv.freg<f$>) -> ()
  // CHECK-NEXT: "riscv.fsw"(%{{.*}}, %{{.*}}) {"immediate" = 1 : i32} : (!riscv.reg<x$>, !riscv.freg<f$>) -> ()

  "riscv.jal"() {"immediate" = 1 : i32, "test" = "hello", "comment" = "world"} : () -> ()
  // CHECK-NEXT: "riscv.jal"() {"immediate" = 1 : i32, "test" = "hello", "comment" = "world"} : () -> ()

  // Unconditional Branch Instructions
  "riscv.ret"() : () -> ()
  // CHECK-NEXT: "riscv.ret"() : () -> ()
}) : () -> ()
