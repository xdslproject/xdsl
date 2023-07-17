// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

"builtin.module"() ({
  "riscv.label"() ({
    %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<zero>
    // CHECK:      li zero, 6
    %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<j1>
    // CHECK-NEXT: li j1, 5
    %2 = "riscv.add"(%0, %1) : (!riscv.reg<zero>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: add j2, zero, j1
    %mv = "riscv.mv"(%0) : (!riscv.reg<zero>) -> !riscv.reg<j2>
    // CHECK-NEXT: mv j2, zero

    // RV32I/RV64I: Integer Computational Instructions (Section 2.4)
    // Integer Register-Immediate Instructions
    %addi = "riscv.addi"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: addi j1, j1, 1
    %slti = "riscv.slti"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: slti j1, j1, 1
    %sltiu = "riscv.sltiu"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: sltiu j1, j1, 1
    %andi = "riscv.andi"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: andi j1, j1, 1
    %ori = "riscv.ori"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: ori j1, j1, 1
    %xori = "riscv.xori"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: xori j1, j1, 1
    %slli = "riscv.slli"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: slli j1, j1, 1
    %srli = "riscv.srli"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: srli j1, j1, 1
    %srai = "riscv.srai"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    // CHECK-NEXT: srai j1, j1, 1
    %lui = "riscv.lui"() {"immediate" = 1 : i32}: () -> !riscv.reg<j0>
    // CHECK-NEXT: lui j0, 1
    %auipc = "riscv.auipc"() {"immediate" = 1 : i32}: () -> !riscv.reg<j0>
    // CHECK-NEXT: auipc j0, 1

    // Integer Register-Register Operations
    %add = "riscv.add"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: add j2, j2, j1
    %slt = "riscv.slt"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: slt j2, j2, j1
    %sltu = "riscv.sltu"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: sltu j2, j2, j1
    %and = "riscv.and"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: and j2, j2, j1
    %or = "riscv.or"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: or j2, j2, j1
    %xor = "riscv.xor"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: xor j2, j2, j1
    %sll = "riscv.sll"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: sll j2, j2, j1
    %srl = "riscv.srl"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: srl j2, j2, j1
    %sub = "riscv.sub"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: sub j2, j2, j1
    %sra = "riscv.sra"(%2, %1) : (!riscv.reg<j2>, !riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: sra j2, j2, j1
    "riscv.nop"() : () -> ()
    // CHECK-NEXT: nop

    // RV32I/RV64I: 2.5 Control Transfer Instructions

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
    "riscv.jalr"(%0) {"immediate" = 1 : i32, "rd" = !riscv.reg<j0>} : (!riscv.reg<zero>) -> ()
    // CHECK-NEXT: jalr j0, zero, 1
    "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<zero>) -> ()
    // CHECK-NEXT: jalr zero, label

    "riscv.ret"() : () -> ()
    // CHECK-NEXT: ret
  ^0(%b00 : !riscv.reg<>, %b01 : !riscv.reg<>):


    // Conditional Branch Instructions
    "riscv.beq"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: beq j2, j1, 1
    "riscv.bne"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: bne j2, j1, 1
    "riscv.blt"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: blt j2, j1, 1
    "riscv.bge"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: bge j2, j1, 1
    "riscv.bltu"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: bltu j2, j1, 1
    "riscv.bgeu"(%2, %1) {"offset" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: bgeu j2, j1, 1

    // RV32I/RV64I: Load and Store Instructions (Section 2.6)
    %lb = "riscv.lb"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: lb j2, j1, 1
    %lbu = "riscv.lbu"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: lbu j2, j1, 1
    %lh = "riscv.lh"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: lh j2, j1, 1
    %lhu = "riscv.lhu"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: lhu j2, j1, 1
    %lw = "riscv.lw"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j2>
    // CHECK-NEXT: lw j2, 1(j1)

    "riscv.sb"(%2, %1) {"immediate" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: sb j2, j1, 1
    "riscv.sh"(%2, %1) {"immediate" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: sh j2, j1, 1
    "riscv.sw"(%2, %1) {"immediate" = 1 : i32}: (!riscv.reg<j2>, !riscv.reg<j1>) -> ()
    // CHECK-NEXT: sw j1, 1(j2)

    // RV32I/RV64I: Control and Status Register Instructions (Section 2.8)
    %csrrw_rw = "riscv.csrrw"(%2) {"csr" = 1024 : i32}: (!riscv.reg<j2>) -> !riscv.reg<j1>
    // CHECK-NEXT: csrrw j1, 1024, j2
    %csrrw_w = "riscv.csrrw"(%2) {"csr" = 1024 : i32, "writeonly"}: (!riscv.reg<j2>) -> !riscv.reg<zero>
    // CHECK-NEXT: csrrw zero, 1024, j2
    %csrrs_rw = "riscv.csrrs"(%2) {"csr" = 1024 : i32}: (!riscv.reg<j2>) -> !riscv.reg<zero>
    // CHECK-NEXT: csrrs zero, 1024, j2
    %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<zero>) -> !riscv.reg<j2>
    // CHECK-NEXT: csrrs j2, 1024, zero
    %csrrc_rw = "riscv.csrrc"(%2) {"csr" = 1024 : i32}: (!riscv.reg<j2>) -> !riscv.reg<j0>
    // CHECK-NEXT: csrrc j0, 1024, j2
    %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<zero>) -> !riscv.reg<j0>
    // CHECK-NEXT: csrrc j0, 1024, zero
    %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<j1>
    // CHECK-NEXT: csrrsi j1, 1024, 8
    %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<j0>
    // CHECK-NEXT: csrrsi j0, 1024, 0
    %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<j0>
    // CHECK-NEXT: csrrci j0, 1024, 8
    %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<j1>
    // CHECK-NEXT: csrrci j1, 1024, 0
    %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32}: () -> !riscv.reg<j0>
    // CHECK-NEXT: csrrwi j0, 1024
    %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly"}: () -> !riscv.reg<zero>
    // CHECK-NEXT: csrrwi zero, 1024

    // Assembler pseudo-instructions
    %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<j0>
    // CHECK-NEXT: li j0, 1
    // Environment Call and Breakpoints
    "riscv.ecall"() : () -> ()
    // CHECK-NEXT: ecall
    "riscv.ebreak"() : () -> ()
    // CHECK-NEXT: ebreak
    "riscv.ret"() : () -> ()
    // CHECK-NEXT: ret
  ^1(%b10 : !riscv.reg<>, %b11 : !riscv.reg<>):

    "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
    // CHECK-NEXT: .align 2
    "riscv.directive"() ({
      %nested_addi = "riscv.addi"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
    }) {"directive" = ".text"} : () -> ()
    // CHECK-NEXT:  .text
    // CHECK-NEXT:  addi j1, j1, 1
    "riscv.label"() {"label" = #riscv.label<"label0">} : () -> ()
    // CHECK-NEXT: label0:
    "riscv.label"() ({
      %nested_addi = "riscv.addi"(%1) {"immediate" = 1 : i32}: (!riscv.reg<j1>) -> !riscv.reg<j1>
      "riscv.ret"() : () -> ()
    }) {"label" = #riscv.label<"label1">} : () -> ()
    // CHECK-NEXT: label1:
    // CHECK-NEXT: addi j1, j1, 1
    // CHECK-NEXT: ret


    // Custom instruction
    %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.reg<zero>, !riscv.reg<j1>) -> (!riscv.reg<j3>, !riscv.reg<j4>)
    // CHECK-NEXT:   hello j3, j4, zero, j1


    // RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0

    %f0 = "riscv.fcvt.s.w"(%0) : (!riscv.reg<zero>) -> !riscv.freg<j5>
    // CHECK-NEXT: fcvt.s.w j5, zero
    %f1 = "riscv.fcvt.s.wu"(%1) : (!riscv.reg<j1>) -> !riscv.freg<j6>
    // CHECK-NEXT: fcvt.s.wu j6, j1
    %f2 = "riscv.fcvt.s.wu"(%1) : (!riscv.reg<j1>) -> !riscv.freg<j7>
    // CHECK-NEXT: fcvt.s.wu j7, j1
    %fmadd = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg<j5>, !riscv.freg<j6>, !riscv.freg<j7>) -> !riscv.freg<j8>
    // CHECK-NEXT: fmadd.s j8, j5, j6, j7
    %fmsub = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg<j5>, !riscv.freg<j6>, !riscv.freg<j7>) -> !riscv.freg<j8>
    // CHECK-NEXT: fmsub.s j8, j5, j6, j7
    %fnmsub = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg<j5>, !riscv.freg<j6>, !riscv.freg<j7>) -> !riscv.freg<j8>
    // CHECK-NEXT: fnmsub.s j8, j5, j6, j7
    %fnmadd = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg<j5>, !riscv.freg<j6>, !riscv.freg<j7>) -> !riscv.freg<j8>
    // CHECK-NEXT: fnmadd.s j8, j5, j6, j7
    %fadd = "riscv.fadd.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fadd.s j8, j5, j6
    %fsub = "riscv.fsub.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fsub.s j8, j5, j6
    %fmul = "riscv.fmul.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fmul.s j8, j5, j6
    %fdiv = "riscv.fdiv.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fdiv.s j8, j5, j6
    %fsqrt = "riscv.fsqrt.s"(%f0) : (!riscv.freg<j5>) -> !riscv.freg<j8>
    // CHECK-NEXT: fsqrt.s j8, j5
    %fsgnj = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fsgnj.s j8, j5, j6
    %fsgnjn = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fsgnjn.s j8, j5, j6
    %fsgnjx = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fsgnjx.s j8, j5, j6
    %fmin = "riscv.fmin.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fmin.s j8, j5, j6
    %fmax = "riscv.fmax.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.freg<j8>
    // CHECK-NEXT: fmax.s j8, j5, j6
    %fcvtws = "riscv.fcvt.w.s"(%f0) : (!riscv.freg<j5>) -> !riscv.reg<j8>
    // CHECK-NEXT: fcvt.w.s j8, j5
    %fcvtwus = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg<j5>) -> !riscv.reg<j8>
    // CHECK-NEXT: fcvt.wu.s j8, j5
    %fmvxw = "riscv.fmv.x.w"(%f0) : (!riscv.freg<j5>) -> !riscv.reg<j8>
    // CHECK-NEXT: fmv.x.w j8, j5
    %feq = "riscv.feq.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.reg<j8>
    // CHECK-NEXT: feq.s j8, j5, j6
    %flt = "riscv.flt.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.reg<j8>
    // CHECK-NEXT: flt.s j8, j5, j6
    %fle = "riscv.fle.s"(%f0, %f1) : (!riscv.freg<j5>, !riscv.freg<j6>) -> !riscv.reg<j8>
    // CHECK-NEXT: fle.s j8, j5, j6
    %fclass = "riscv.fclass.s"(%f0) : (!riscv.freg<j5>) -> !riscv.reg<j8>
    // CHECK-NEXT: fclass.s j8, j5
    %fcvtsw = "riscv.fcvt.s.w"(%0) : (!riscv.reg<zero>) -> !riscv.freg<j8>
    // CHECK-NEXT: fcvt.s.w j8, zero
    %fcvtswu = "riscv.fcvt.s.wu"(%0) : (!riscv.reg<zero>) -> !riscv.freg<j8>
    // CHECK-NEXT: fcvt.s.wu j8, zero
    %fmvwx = "riscv.fmv.w.x"(%0) : (!riscv.reg<zero>) -> !riscv.freg<j8>
    // CHECK-NEXT: fmv.w.x j8, zero
    %flw = "riscv.flw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<zero>) -> !riscv.freg<j8>
    // CHECK-NEXT: flw j8, zero, 1
    "riscv.fsw"(%0, %f0) {"immediate" = 1 : i32} : (!riscv.reg<zero>, !riscv.freg<j5>) -> ()
    // CHECK-NEXT: fsw zero, j5, 1

    // Terminate block
    "riscv.ret"() : () -> ()
    // CHECK-NEXT: ret
  }) {"label" = #riscv.label<"main">}: () -> ()
}) : () -> ()
