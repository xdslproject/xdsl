// RUN: xdsl-opt %s | filecheck %s
"builtin.module"() ({
  %0 = "riscv.get_register"() : () -> !riscv.reg<>
  %1 = "riscv.get_register"() : () -> !riscv.reg<>
  // CHECK: %{{.*}} = riscv.get_register : () -> (!riscv.reg<>)
  // CHECK-NEXT: %{{.*}} = riscv.get_register : () -> (!riscv.reg<>)
  // RV32I/RV64I: 2.4 Integer Computational Instructions

  // Integer Register-Immediate Instructions
  %addi = "riscv.addi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %slti = "riscv.slti"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %sltiu = "riscv.sltiu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %andi = "riscv.andi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %ori = "riscv.ori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %xori = "riscv.xori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %slli = "riscv.slli"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %srli = "riscv.srli"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %srai = "riscv.srai"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %lui = "riscv.lui"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  %auipc = "riscv.auipc"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  %mv = "riscv.mv"(%0) : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %addi = riscv.addi %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %slti = riscv.slti %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %sltiu = riscv.sltiu %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %andi = riscv.andi %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %ori = riscv.ori %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %xori = riscv.xori %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %slli = riscv.slli %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %srli = riscv.srli %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %srai = riscv.srai %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %lui = riscv.lui 1 : () -> (!riscv.reg<>)
  // CHECK-NEXT: %auipc = riscv.auipc 1 : () -> (!riscv.reg<>)
  // CHECK-NEXT: %mv = riscv.mv %0 : (!riscv.reg<>) -> (!riscv.reg<>)

  // Integer Register-Register Operations
  %add = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %slt = "riscv.slt"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %sltu = "riscv.sltu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %and = "riscv.and"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %or = "riscv.or"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %xor = "riscv.xor"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %sll = "riscv.sll"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %srl = "riscv.srl"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %sub = "riscv.sub"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %sra = "riscv.sra"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  "riscv.nop"() : () -> ()
  // CHECK-NEXT: %add = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %slt = riscv.slt %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %sltu = riscv.sltu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %and = riscv.and %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %or = riscv.or %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %xor = riscv.xor %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %sll = riscv.sll %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %srl = riscv.srl %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %sub = riscv.sub %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %sra = riscv.sra %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: riscv.nop : () -> ()

  // RV32I/RV64I: 2.5 Control Transfer Instructions
  // terminators continue at the end of module

  // Unconditional Branch Instructions
  "riscv.jal"() {"immediate" = 1 : i32} : () -> ()
  "riscv.jal"() {"immediate" = 1 : i32, "rd" = !riscv.reg<>} : () -> ()
  "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: riscv.jal 1 : () -> ()
  // CHECK-NEXT: riscv.jal 1, None : () -> ()
  // CHECK-NEXT: riscv.jal "label" : () -> ()

  "riscv.j"() {"immediate" = 1 : i32} : () -> ()
  "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: riscv.j 1 : () -> ()
  // CHECK-NEXT: riscv.j "label" : () -> ()

  "riscv.jalr"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> ()
  "riscv.jalr"(%0) {"immediate" = 1 : i32, "rd" = !riscv.reg<>} : (!riscv.reg<>) -> ()
  "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.jalr %0, 1 : (!riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.jalr %0, 1, None : (!riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.jalr %0, "label" : (!riscv.reg<>) -> ()

  // Conditional Branch Instructions
  "riscv.beq"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.bne"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.blt"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.bge"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.bltu"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.bgeu"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.beq %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.bne %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.blt %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.bge %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.bltu %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.bgeu %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()

  // RV32I/RV64I: 2.6 Load and Store Instructions

  %lb = "riscv.lb"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %lbu = "riscv.lbu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %lh = "riscv.lh"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %lhu = "riscv.lhu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %lw = "riscv.lw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  "riscv.sb"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.sh"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  "riscv.sw"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: %lb = riscv.lb %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %lbu = riscv.lbu %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %lh = riscv.lh %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %lhu = riscv.lhu %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %lw = riscv.lw %0, 1 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: riscv.sb %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.sh %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.sw %0, %1, 1 : (!riscv.reg<>, !riscv.reg<>) -> ()

  // RV32I/RV64I: 2.8 Control and Status Register Instructions

  %csrrw_rw = "riscv.csrrw"(%0) {"csr" = 1024 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %csrrw_w = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"}: (!riscv.reg<>) -> !riscv.reg<>
  %csrrs_rw = "riscv.csrrs"(%0) {"csr" = 1024 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<>) -> !riscv.reg<>
  %csrrc_rw = "riscv.csrrc"(%0) {"csr" = 1024 : i32}: (!riscv.reg<>) -> !riscv.reg<>
  %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<>) -> !riscv.reg<>
  %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<>
  %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<>
  %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<>
  %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<>
  %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32}: () -> !riscv.reg<>
  %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %csrrw_rw = riscv.csrrw %0, 1024 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrw_w = riscv.csrrw %0, 1024, w : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrs_rw = riscv.csrrs %0, 1024 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrs_r = riscv.csrrs %0, 1024, r : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrc_rw = riscv.csrrc %0, 1024 : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrc_r = riscv.csrrc %0, 1024, r : (!riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrsi_rw = riscv.csrrsi 1024, 8 : () -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrsi_r = riscv.csrrsi 1024, 0 : () -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrci_rw = riscv.csrrci 1024, 8 : () -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrci_r = riscv.csrrci 1024, 0 : () -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrwi_rw = riscv.csrrwi 1024, 1 : () -> (!riscv.reg<>)
  // CHECK-NEXT: %csrrwi_w = riscv.csrrwi 1024, w, 1 : () -> (!riscv.reg<>)

  // Machine Mode Privileged Instructions
  "riscv.wfi"() : () -> ()
  // CHECK-NEXT: riscv.wfi : () -> ()


  // RV32M/RV64M: 7 “M” Standard Extension for Integer Multiplication and Division

  // Multiplication Operations
  %mul = "riscv.mul"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %mulh = "riscv.mulh"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %mulhsu = "riscv.mulhsu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %mulhu = "riscv.mulhu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %mul = riscv.mul %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %mulh = riscv.mulh %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %mulhsu = riscv.mulhsu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %mulhu = riscv.mulhu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)

  // Division Operations
  %div = "riscv.div"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %divu = "riscv.divu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %rem = "riscv.rem"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  %remu = "riscv.remu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %div = riscv.div %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %divu = riscv.divu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %rem = riscv.rem %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %remu = riscv.remu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>)

  // Assembler pseudo-instructions

  %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<>
  // CHECK-NEXT: %li = riscv.li 1 : () -> (!riscv.reg<>)

  // Environment Call and Breakpoints
  "riscv.ecall"() : () -> ()
  "riscv.ebreak"() : () -> ()
  "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
  "riscv.directive"() ({
    %nested_li = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
  }) {"directive" = ".text"} : () -> ()
  // CHECK-NEXT: riscv.ecall : () -> ()
  // CHECK-NEXT: riscv.ebreak : () -> ()
  // CHECK-NEXT: riscv.directive ".align", "2" : () -> ()
  // CHECK-NEXT: riscv.directive ".text" ({
  // CHECK-NEXT:   %nested_li = riscv.li 1 : () -> (!riscv.reg<>)
  // CHECK-NEXT: }) : () -> ()

  // Custom instruction
  %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>, !riscv.reg<>)
  // CHECK-NEXT: %custom0, %custom1 = riscv.custom_assembly_instruction "hello" %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>, !riscv.reg<>)


  // RISC-V extensions
  "riscv.scfgw"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: riscv.scfgw %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> ()

  // RV32I/RV64I: 2.5 Control Transfer Instructions (cont'd)
  // terminators

  // RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0
  %f0 = "riscv.get_float_register"() : () -> !riscv.freg<>
  %f1 = "riscv.get_float_register"() : () -> !riscv.freg<>
  %f2 = "riscv.get_float_register"() : () -> !riscv.freg<>
  // CHECK-NEXT: %f0 = riscv.get_float_register : () -> (!riscv.freg<>)
  // CHECK-NEXT: %f1 = riscv.get_float_register : () -> (!riscv.freg<>)
  // CHECK-NEXT: %f2 = riscv.get_float_register : () -> (!riscv.freg<>)

  %fmadd_s = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fmsub_s = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fnmsub_s = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fnmadd_s = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %fmadd_s = riscv.fmadd.s %f0, %f1, %f2 : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fmsub_s = riscv.fmsub.s %f0, %f1, %f2 : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fnmsub_s = riscv.fnmsub.s %f0, %f1, %f2 : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fnmadd_s = riscv.fnmadd.s %f0, %f1, %f2 : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)

  %fadd_s = "riscv.fadd.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fsub_s = "riscv.fsub.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fmul_s = "riscv.fmul.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fdiv_s = "riscv.fdiv.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fsqrt_s = "riscv.fsqrt.s"(%f0) : (!riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %fadd_s = riscv.fadd.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fsub_s = riscv.fsub.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fmul_s = riscv.fmul.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fdiv_s = riscv.fdiv.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fsqrt_s = riscv.fsqrt.s %f0 : (!riscv.freg<>) -> (!riscv.freg<>)

  %fsgnj_s = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fsgnjn_s = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fsgnjx_s = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %fsgnj_s = riscv.fsgnj.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fsgnjn_s = riscv.fsgnjn.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fsgnjx_s = riscv.fsgnjx.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)

  %fmin_s = "riscv.fmin.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %fmax_s = "riscv.fmax.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  // CHECK-NEXT: %fmin_s = riscv.fmin.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fmax_s = riscv.fmax.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.freg<>)

  %fcvt_w_s = "riscv.fcvt.w.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
  %fcvt_wu_s = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
  %fmv_x_w = "riscv.fmv.x.w"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
  // CHECK-NEXT: %fcvt_w_s = riscv.fcvt.w.s %f0 : (!riscv.freg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %fcvt_wu_s = riscv.fcvt.wu.s %f0 : (!riscv.freg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %fmv_x_w = riscv.fmv.x.w %f0 : (!riscv.freg<>) -> (!riscv.reg<>)

  %feq_s = "riscv.feq.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
  %flt_s = "riscv.flt.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
  %fle_s = "riscv.fle.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
  %fclass_s = "riscv.fclass.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
  %fcvt_s_w = "riscv.fcvt.s.w"(%0) : (!riscv.reg<>) -> !riscv.freg<>
  %fcvt_s_wu = "riscv.fcvt.s.wu"(%0) : (!riscv.reg<>) -> !riscv.freg<>
  %fmv_w_x = "riscv.fmv.w.x"(%0) : (!riscv.reg<>) -> !riscv.freg<>
  // CHECK-NEXT: %feq_s = riscv.feq.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %flt_s = riscv.flt.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %fle_s = riscv.fle.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %fclass_s = riscv.fclass.s %f0 : (!riscv.freg<>) -> (!riscv.reg<>)
  // CHECK-NEXT: %fcvt_s_w = riscv.fcvt.s.w %0 : (!riscv.reg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fcvt_s_wu = riscv.fcvt.s.wu %0 : (!riscv.reg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: %fmv_w_x = riscv.fmv.w.x %0 : (!riscv.reg<>) -> (!riscv.freg<>)

  %flw = "riscv.flw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<>) -> !riscv.freg<>
  "riscv.fsw"(%0, %f0) {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.freg<>) -> ()
  // CHECK-NEXT: %flw = riscv.flw %0, 1 : (!riscv.reg<>) -> (!riscv.freg<>)
  // CHECK-NEXT: riscv.fsw %0, %f0, 1 : (!riscv.reg<>, !riscv.freg<>) -> ()

  "riscv.jal"() {"immediate" = 1 : i32, "test" = "hello", "comment" = "world"} : () -> ()
  // CHECK-NEXT: riscv.jal 1 {"test" = "hello"} : () -> ()

  // Unconditional Branch Instructions
  "riscv.ret"() : () -> ()
  // CHECK-NEXT: riscv.ret : () -> ()
}) : () -> ()
