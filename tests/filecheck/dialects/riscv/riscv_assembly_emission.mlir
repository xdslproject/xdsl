// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

"builtin.module"() ({
  %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<zero>
  // CHECK:      li zero, 6
  %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<jx1>
  // CHECK-NEXT: li jx1, 5
  %2 = "riscv.add"(%0, %1) : (!riscv.reg<zero>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: add jx2, zero, jx1
  %mv = "riscv.mv"(%0) : (!riscv.reg<zero>) -> !riscv.reg<jx2>
  // CHECK-NEXT: mv jx2, zero

  // RV32I/RV64I: Integer Computational Instructions (Section 2.4)
  // Integer Register-Immediate Instructions
  %addi = "riscv.addi"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: addi jx1, jx1, 1
  %slti = "riscv.slti"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: slti jx1, jx1, 1
  %sltiu = "riscv.sltiu"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: sltiu jx1, jx1, 1
  %andi = "riscv.andi"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: andi jx1, jx1, 1
  %ori = "riscv.ori"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: ori jx1, jx1, 1
  %xori = "riscv.xori"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: xori jx1, jx1, 1
  %slli = "riscv.slli"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: slli jx1, jx1, 1
  %srli = "riscv.srli"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: srli jx1, jx1, 1
  %srai = "riscv.srai"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  // CHECK-NEXT: srai jx1, jx1, 1
  %lui = "riscv.lui"() {"immediate" = 1 : i32}: () -> !riscv.reg<jx0>
  // CHECK-NEXT: lui jx0, 1
  %auipc = "riscv.auipc"() {"immediate" = 1 : i32}: () -> !riscv.reg<jx0>
  // CHECK-NEXT: auipc jx0, 1

  // Integer Register-Register Operations
  %add = "riscv.add"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: add jx2, jx2, jx1
  %slt = "riscv.slt"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: slt jx2, jx2, jx1
  %sltu = "riscv.sltu"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: sltu jx2, jx2, jx1
  %and = "riscv.and"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: and jx2, jx2, jx1
  %or = "riscv.or"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: or jx2, jx2, jx1
  %xor = "riscv.xor"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: xor jx2, jx2, jx1
  %sll = "riscv.sll"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: sll jx2, jx2, jx1
  %srl = "riscv.srl"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: srl jx2, jx2, jx1
  %sub = "riscv.sub"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: sub jx2, jx2, jx1
  %sra = "riscv.sra"(%2, %1) : (!riscv.reg<jx2>, !riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: sra jx2, jx2, jx1
  "riscv.nop"() : () -> ()
  // CHECK-NEXT: nop

  // RV32I/RV64I: 2.5 Control Transfer Instructions
  // terminators continue at the end of module

  // Unconditional Branch Instructions
  "riscv.jal"() {"immediate" = 1 : i32} : () -> ()
  // CHECK-NEXT: jal 1
  "riscv.jal"() {"immediate" = 1 : i32, "rd" = !riscv.reg<s0>} : () -> ()
  // CHECK-NEXT: jal s0, 1
  "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: jal label

  "riscv.j"() {"immediate" = 1 : i32, "rd" = !riscv.reg<zero>} : () -> ()
  // CHECK-NEXT: j 1
  "riscv.j"() {"immediate" = #riscv.label<"label">, "rd" = !riscv.reg<zero>} : () -> ()
  // CHECK-NEXT: j label

  "riscv.jalr"(%0) {"immediate" = 1 : i32}: (!riscv.reg<zero>) -> ()
  // CHECK-NEXT: jalr zero, 1
  "riscv.jalr"(%0) {"immediate" = 1 : i32, "rd" = !riscv.reg<jx0>} : (!riscv.reg<zero>) -> ()
  // CHECK-NEXT: jalr jx0, zero, 1
  "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<zero>) -> ()
  // CHECK-NEXT: jalr zero, label

  // Conditional Branch Instructions
  "riscv.beq"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: beq jx2, jx1, 1
  "riscv.bne"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: bne jx2, jx1, 1
  "riscv.blt"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: blt jx2, jx1, 1
  "riscv.bge"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: bge jx2, jx1, 1
  "riscv.bltu"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: bltu jx2, jx1, 1
  "riscv.bgeu"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: bgeu jx2, jx1, 1

  // RV32I/RV64I: Load and Store Instructions (Section 2.6)
  %lb = "riscv.lb"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: lb jx2, jx1, 1
  %lbu = "riscv.lbu"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: lbu jx2, jx1, 1
  %lh = "riscv.lh"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: lh jx2, jx1, 1
  %lhu = "riscv.lhu"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: lhu jx2, jx1, 1
  %lw = "riscv.lw"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx2>
  // CHECK-NEXT: lw jx2, 1(jx1)

  "riscv.sb"(%2, %1) {"immediate" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: sb jx2, jx1, 1
  "riscv.sh"(%2, %1) {"immediate" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: sh jx2, jx1, 1
  "riscv.sw"(%2, %1) {"immediate" = 1 : i32}: (!riscv.reg<jx2>, !riscv.reg<jx1>) -> ()
  // CHECK-NEXT: sw jx1, 1(jx2)

  // RV32I/RV64I: Control and Status Register Instructions (Section 2.8)
  %csrrw_rw = "riscv.csrrw"(%2) {"csr" = 1024 : i32}: (!riscv.reg<jx2>) -> !riscv.reg<jx1>
  // CHECK-NEXT: csrrw jx1, 1024, jx2
  %csrrw_w = "riscv.csrrw"(%2) {"csr" = 1024 : i32, "writeonly"}: (!riscv.reg<jx2>) -> !riscv.reg<zero>
  // CHECK-NEXT: csrrw zero, 1024, jx2
  %csrrs_rw = "riscv.csrrs"(%2) {"csr" = 1024 : i32}: (!riscv.reg<jx2>) -> !riscv.reg<zero>
  // CHECK-NEXT: csrrs zero, 1024, jx2
  %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<zero>) -> !riscv.reg<jx2>
  // CHECK-NEXT: csrrs jx2, 1024, zero
  %csrrc_rw = "riscv.csrrc"(%2) {"csr" = 1024 : i32}: (!riscv.reg<jx2>) -> !riscv.reg<jx0>
  // CHECK-NEXT: csrrc jx0, 1024, jx2
  %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<zero>) -> !riscv.reg<jx0>
  // CHECK-NEXT: csrrc jx0, 1024, zero
  %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<jx1>
  // CHECK-NEXT: csrrsi jx1, 1024, 8
  %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<jx0>
  // CHECK-NEXT: csrrsi jx0, 1024, 0
  %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<jx0>
  // CHECK-NEXT: csrrci jx0, 1024, 8
  %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<jx1>
  // CHECK-NEXT: csrrci jx1, 1024, 0
  %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32}: () -> !riscv.reg<jx0>
  // CHECK-NEXT: csrrwi jx0, 1024
  %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly"}: () -> !riscv.reg<zero>
  // CHECK-NEXT: csrrwi zero, 1024

  // Assembler pseudo-instructions
  %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<jx0>
  // CHECK-NEXT: li jx0, 1
  // Environment Call and Breakpoints
  "riscv.ecall"() : () -> ()
  // CHECK-NEXT: ecall
  "riscv.ebreak"() : () -> ()
  // CHECK-NEXT: ebreak
  "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
  // CHECK-NEXT: .align 2
  "riscv.directive"() ({
    %nested_addi = "riscv.addi"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  }) {"directive" = ".text"} : () -> ()
  // CHECK-NEXT:  .text
  // CHECK-NEXT:  addi jx1, jx1, 1
  "riscv.label"() {"label" = #riscv.label<"label0">} : () -> ()
  // CHECK-NEXT: label0:
  "riscv.label"() ({
    %nested_addi = "riscv.addi"(%1) {"immediate" = 1 : i32}: (!riscv.reg<jx1>) -> !riscv.reg<jx1>
  }) {"label" = #riscv.label<"label1">} : () -> ()
  // CHECK-NEXT: label1:
  // CHECK-NEXT: addi jx1, jx1, 1


  // Custom instruction
  %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.reg<zero>, !riscv.reg<jx1>) -> (!riscv.reg<jx3>, !riscv.reg<jx4>)
  // CHECK-NEXT:   hello jx3, jx4, zero, jx1

  // RV32I/RV64I: 2.5 Control Transfer Instructions (cont'd)
  // terminators

  // RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0
  %f0 = "riscv.fcvt.s.w"(%0) : (!riscv.reg<zero>) -> !riscv.freg<jf5>
  // CHECK-NEXT: fcvt.s.w jf5, zero
  %f1 = "riscv.fcvt.s.wu"(%1) : (!riscv.reg<jx1>) -> !riscv.freg<jf6>
  // CHECK-NEXT: fcvt.s.wu jf6, jx1
  %f2 = "riscv.fcvt.s.wu"(%1) : (!riscv.reg<jx1>) -> !riscv.freg<jf7>
  // CHECK-NEXT: fcvt.s.wu jf7, jx1
  %fmadd = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg<jf5>, !riscv.freg<jf6>, !riscv.freg<jf7>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fmadd.s jf8, jf5, jf6, jf7
  %fmsub = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg<jf5>, !riscv.freg<jf6>, !riscv.freg<jf7>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fmsub.s jf8, jf5, jf6, jf7
  %fnmsub = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg<jf5>, !riscv.freg<jf6>, !riscv.freg<jf7>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fnmsub.s jf8, jf5, jf6, jf7
  %fnmadd = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg<jf5>, !riscv.freg<jf6>, !riscv.freg<jf7>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fnmadd.s jf8, jf5, jf6, jf7
  %fadd = "riscv.fadd.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fadd.s jf8, jf5, jf6
  %fsub = "riscv.fsub.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fsub.s jf8, jf5, jf6
  %fmul = "riscv.fmul.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fmul.s jf8, jf5, jf6
  %fdiv = "riscv.fdiv.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fdiv.s jf8, jf5, jf6
  %fsqrt = "riscv.fsqrt.s"(%f0) : (!riscv.freg<jf5>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fsqrt.s jf8, jf5
  %fsgnj = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fsgnj.s jf8, jf5, jf6
  %fsgnjn = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fsgnjn.s jf8, jf5, jf6
  %fsgnjx = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fsgnjx.s jf8, jf5, jf6
  %fmin = "riscv.fmin.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fmin.s jf8, jf5, jf6
  %fmax = "riscv.fmax.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fmax.s jf8, jf5, jf6
  %fcvtws = "riscv.fcvt.w.s"(%f0) : (!riscv.freg<jf5>) -> !riscv.reg<jx8>
  // CHECK-NEXT: fcvt.w.s jx8, jf5
  %fcvtwus = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg<jf5>) -> !riscv.reg<jx8>
  // CHECK-NEXT: fcvt.wu.s jx8, jf5
  %fmvxw = "riscv.fmv.x.w"(%f0) : (!riscv.freg<jf5>) -> !riscv.reg<jx8>
  // CHECK-NEXT: fmv.x.w jx8, jf5
  %feq = "riscv.feq.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.reg<jx8>
  // CHECK-NEXT: feq.s jx8, jf5, jf6
  %flt = "riscv.flt.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.reg<jx8>
  // CHECK-NEXT: flt.s jx8, jf5, jf6
  %fle = "riscv.fle.s"(%f0, %f1) : (!riscv.freg<jf5>, !riscv.freg<jf6>) -> !riscv.reg<jx8>
  // CHECK-NEXT: fle.s jx8, jf5, jf6
  %fclass = "riscv.fclass.s"(%f0) : (!riscv.freg<jf5>) -> !riscv.reg<jx8>
  // CHECK-NEXT: fclass.s jx8, jf5
  %fcvtsw = "riscv.fcvt.s.w"(%0) : (!riscv.reg<zero>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fcvt.s.w jf8, zero
  %fcvtswu = "riscv.fcvt.s.wu"(%0) : (!riscv.reg<zero>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fcvt.s.wu jf8, zero
  %fmvwx = "riscv.fmv.w.x"(%0) : (!riscv.reg<zero>) -> !riscv.freg<jf8>
  // CHECK-NEXT: fmv.w.x jf8, zero
  %flw = "riscv.flw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<zero>) -> !riscv.freg<jf8>
  // CHECK-NEXT: flw jf8, zero, 1
  "riscv.fsw"(%0, %f0) {"immediate" = 1 : i32} : (!riscv.reg<zero>, !riscv.freg<jf5>) -> ()
  // CHECK-NEXT: fsw zero, jf5, 1

  // Unconditional Branch Instructions
  "riscv.ret"() : () -> () // pseudo-instruction
  // CHECK-NEXT: ret
}) : () -> ()
