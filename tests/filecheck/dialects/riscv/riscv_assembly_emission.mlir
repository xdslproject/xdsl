// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

"builtin.module"() ({
  riscv_func.func @main() {
    %0 = riscv.li 6 : !riscv.reg<zero>
    // CHECK:      li zero, 6
    %1 = riscv.li 5 : !riscv.reg<j_1>
    // CHECK-NEXT: li j_1, 5
    %2 = riscv.add %0, %1 : (!riscv.reg<zero>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: add j_2, zero, j_1
    %mv = riscv.mv %0 : (!riscv.reg<zero>) -> !riscv.reg<j_2>
    // CHECK-NEXT: mv j_2, zero
    %seqz = riscv.seqz %1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: seqz j_1, j_1
    %snez = riscv.snez %1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: snez j_1, j_1
    %zextb = riscv.zext.b %1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: zext.b j_1, j_1
    %zextw = riscv.zext.w %1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: zext.w j_1, j_1
    %sextw = riscv.sext.w %1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: sext.w j_1, j_1

    // RV32I/RV64I: Integer Computational Instructions (Section 2.4)
    // Integer Register-Immediate Instructions
    %addi = riscv.addi %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: addi j_1, j_1, 1
    %slti = riscv.slti %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: slti j_1, j_1, 1
    %sltiu = riscv.sltiu %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: sltiu j_1, j_1, 1
    %andi = riscv.andi %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: andi j_1, j_1, 1
    %ori = riscv.ori %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: ori j_1, j_1, 1
    %xori = riscv.xori %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: xori j_1, j_1, 1
    %slli = riscv.slli %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: slli j_1, j_1, 1
    %srli = riscv.srli %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: srli j_1, j_1, 1
    %srai = riscv.srai %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: srai j_1, j_1, 1
    %lui = riscv.lui 1: () -> !riscv.reg<j_0>
    // CHECK-NEXT: lui j_0, 1
    %auipc = riscv.auipc 1: () -> !riscv.reg<j_0>
    // CHECK-NEXT: auipc j_0, 1

    // Integer Register-Register Operations
    %add = riscv.add %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: add j_2, j_2, j_1
    %slt = riscv.slt %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: slt j_2, j_2, j_1
    %sltu = riscv.sltu %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sltu j_2, j_2, j_1
    %and = riscv.and %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: and j_2, j_2, j_1
    %or = riscv.or %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: or j_2, j_2, j_1
    %xor = riscv.xor %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: xor j_2, j_2, j_1
    %sll = riscv.sll %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sll j_2, j_2, j_1
    %srl = riscv.srl %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: srl j_2, j_2, j_1
    %sub = riscv.sub %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sub j_2, j_2, j_1
    %sra = riscv.sra %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sra j_2, j_2, j_1
    riscv.nop
    // CHECK-NEXT: nop

    // RV32I/RV64I: 2.5 Control Transfer Instructions

    // Unconditional Branch Instructions
    riscv.jal 1
    // CHECK-NEXT: jal 1
    riscv.jal 1, !riscv.reg<s0>
    // CHECK-NEXT: jal s0, 1
    riscv.jal "label"
    // CHECK-NEXT: jal label

    riscv.j 1, !riscv.reg<zero>
    // CHECK-NEXT: j 1
    riscv.j "label", !riscv.reg<zero>
    // CHECK-NEXT: j label

    riscv.jalr %0, 1 : (!riscv.reg<zero>) -> ()
    // CHECK-NEXT: jalr zero, 1
    riscv.jalr %0 1, !riscv.reg<j_0> : (!riscv.reg<zero>) -> ()
    // CHECK-NEXT: jalr j_0, zero, 1
    riscv.jalr %0 "label" : (!riscv.reg<zero>) -> ()
    // CHECK-NEXT: jalr zero, label

    riscv.ret
    // CHECK-NEXT: ret
  ^bb0(%b00 : !riscv.reg, %b01 : !riscv.reg):


    // Conditional Branch Instructions
    riscv.beq %2, %1, 1: (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: beq j_2, j_1, 1
    riscv.bne %2, %1, 1: (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: bne j_2, j_1, 1
    riscv.blt %2, %1, 1: (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: blt j_2, j_1, 1
    riscv.bge %2, %1, 1: (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: bge j_2, j_1, 1
    riscv.bltu %2, %1, 1: (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: bltu j_2, j_1, 1
    riscv.bgeu %2, %1, 1: (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: bgeu j_2, j_1, 1

    // RV32I/RV64I: Load and Store Instructions (Section 2.6)
    %lb = riscv.lb %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: lb j_2, j_1, 1
    %lbu = riscv.lbu %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: lbu j_2, j_1, 1
    %lh = riscv.lh %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: lh j_2, j_1, 1
    %lhu = riscv.lhu %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: lhu j_2, j_1, 1
    %lw = riscv.lw %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: lw j_2, 1(j_1)

    riscv.sb %2, %1, 1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: sb j_2, j_1, 1
    riscv.sh %2, %1, 1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: sh j_2, j_1, 1
    riscv.sw %2, %1, 1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> ()
    // CHECK-NEXT: sw j_1, 1(j_2)

    // RV32I/RV64I: Control and Status Register Instructions (Section 2.8)
    %csrrw_rw = riscv.csrrw %2, 1024 : (!riscv.reg<j_2>) -> !riscv.reg<j_1>
    // CHECK-NEXT: csrrw j_1, 1024, j_2
    %csrrw_w = riscv.csrrw %2, 1024, "w" : (!riscv.reg<j_2>) -> !riscv.reg<zero>
    // CHECK-NEXT: csrrw zero, 1024, j_2
    %csrrs_rw = riscv.csrrs %2, 1024 : (!riscv.reg<j_2>) -> !riscv.reg<zero>
    // CHECK-NEXT: csrrs zero, 1024, j_2
    %csrrs_r = riscv.csrrs %0, 1024, "r" : (!riscv.reg<zero>) -> !riscv.reg<j_2>
    // CHECK-NEXT: csrrs j_2, 1024, zero
    %csrrc_rw = riscv.csrrc %2, 1024 : (!riscv.reg<j_2>) -> !riscv.reg<j_0>
    // CHECK-NEXT: csrrc j_0, 1024, j_2
    %csrrc_r = riscv.csrrc %0, 1024, "r": (!riscv.reg<zero>) -> !riscv.reg<j_0>
    // CHECK-NEXT: csrrc j_0, 1024, zero
    %csrrsi_rw = riscv.csrrsi 1024, 8 : () -> !riscv.reg<j_1>
    // CHECK-NEXT: csrrsi j_1, 1024, 8
    %csrrsi_r = riscv.csrrsi 1024, 0 : () -> !riscv.reg<j_0>
    // CHECK-NEXT: csrrsi j_0, 1024, 0
    %csrrci_rw = riscv.csrrci 1024, 8 : () -> !riscv.reg<j_0>
    // CHECK-NEXT: csrrci j_0, 1024, 8
    %csrrci_r = riscv.csrrci 1024, 0 : () -> !riscv.reg<j_1>
    // CHECK-NEXT: csrrci j_1, 1024, 0
    %csrrwi_rw = riscv.csrrwi 1024, 8 : () -> !riscv.reg<j_0>
    // CHECK-NEXT: csrrwi j_0, 1024, 8
    %csrrwi_w = riscv.csrrwi 1024, 8, "w" : () -> !riscv.reg<zero>
    // CHECK-NEXT: csrrwi zero, 1024, 8

    // Assembler pseudo-instructions
    %li = riscv.li 1: !riscv.reg<j_0>
    // CHECK-NEXT: li j_0, 1
    // Environment Call and Breakpoints
    riscv.ecall
    // CHECK-NEXT: ecall
    riscv.ebreak
    // CHECK-NEXT: ebreak
    riscv.ret
    // CHECK-NEXT: ret
  ^bb1(%b10 : !riscv.reg, %b11 : !riscv.reg):

    riscv.directive ".align" "2"
    // CHECK-NEXT: .align 2
    riscv.assembly_section ".text" {
      %inner = riscv.li 5 : !riscv.reg<j_1>
      %nested_addi = riscv.addi %inner, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    }
    // CHECK-NEXT:  li j_1, 5
    // CHECK-NEXT:  addi j_1, j_1, 1
    riscv.label "label0"
    // CHECK-NEXT: label0:


    // Custom instruction
    %custom0, %custom1 = riscv.custom_assembly_instruction %0, %1 {"instruction_name" = "hello"} : (!riscv.reg<zero>, !riscv.reg<j_1>) -> (!riscv.reg<j_3>, !riscv.reg<j_4>)
    // CHECK-NEXT:   hello j_3, j_4, zero, j_1


    // RISC-V Extensions

    riscv_snitch.frep_outer %0 {
      %add_o = riscv.add %0, %1 : (!riscv.reg<zero>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    }

    // CHECK:          frep.o zero, 1, 0, 0
    // CHECK-NEXT:     add  j_2, zero, j_1

    riscv_snitch.frep_inner %0 {
      %add_i = riscv.add %0, %1 : (!riscv.reg<zero>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    }
    // CHECK:          frep.i zero, 1, 0, 0
    // CHECK-NEXT:     add  j_2, zero, j_1
    
    //RV32B/RV64B: "B" Extension for Bit Manipulation, Version 1.0.0
    %rol = riscv.rol %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: rol j_2, j_2, j_1
    %ror = riscv.ror %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: ror j_2, j_2, j_1
    %rolw = riscv.rolw %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: rolw j_2, j_2, j_1
    %rorw = riscv.rorw %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: rorw j_2, j_2, j_1
    %rori = riscv.rori %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: rori j_1, j_1, 1
    %roriw = riscv.roriw %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: roriw j_1, j_1, 1
    %bclr = riscv.bclr %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: bclr j_2, j_2, j_1
    %bclri = riscv.bclri %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: bclri j_1, j_1, 1
    %bseti = riscv.bseti %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: bseti j_1, j_1, 1
    %adduw = riscv.add.uw %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: add.uw j_2, j_2, j_1
    %sh1add = riscv.sh1add %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sh1add j_2, j_2, j_1
    %sh2add = riscv.sh2add %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sh2add j_2, j_2, j_1
    %sh3add = riscv.sh3add %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sh3add j_2, j_2, j_1
    %sh1adduw = riscv.sh1add.uw %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sh1add.uw j_2, j_2, j_1
    %sh2adduw = riscv.sh2add.uw %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sh2add.uw j_2, j_2, j_1
    %sh3adduw = riscv.sh3add.uw %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: sh3add.uw j_2, j_2, j_1
    %slliuw = riscv.slli.uw %1, 1 : (!riscv.reg<j_1>) -> !riscv.reg<j_1>
    // CHECK-NEXT: slli.uw j_1, j_1, 1
    %andn = riscv.andn %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: andn j_2, j_2, j_1
    %orn = riscv.orn %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: orn j_2, j_2, j_1
    %xnor = riscv.xnor %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: xnor j_2, j_2, j_1
    %max = riscv.max %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: max j_2, j_2, j_1
    %maxu = riscv.maxu %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: maxu j_2, j_2, j_1
    %min = riscv.min %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: min j_2, j_2, j_1
    %minu = riscv.minu %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: minu j_2, j_2, j_1
    
    // "ZiCond" Conditional" operations extension
    %czeroeqz = riscv.czero.eqz %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: czero.eqz j_2, j_2, j_1
    %czeronez = riscv.czero.nez %2, %1 : (!riscv.reg<j_2>, !riscv.reg<j_1>) -> !riscv.reg<j_2>
    // CHECK-NEXT: czero.nez j_2, j_2, j_1

    // RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0

    %f0 = riscv.fcvt.s.w %0 : (!riscv.reg<zero>) -> !riscv.freg<fj_0>
    // CHECK-NEXT: fcvt.s.w fj_0, zero
    %f1 = riscv.fcvt.s.wu %1 : (!riscv.reg<j_1>) -> !riscv.freg<fj_1>
    // CHECK-NEXT: fcvt.s.wu fj_1, j_1
    %f2 = riscv.fcvt.s.wu %1 : (!riscv.reg<j_1>) -> !riscv.freg<fj_2>
    // CHECK-NEXT: fcvt.s.wu fj_2, j_1
    %fmadd = riscv.fmadd.s %f0, %f1, %f2 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmadd.s fj_3, fj_0, fj_1, fj_2
    %fmsub = riscv.fmsub.s %f0, %f1, %f2 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmsub.s fj_3, fj_0, fj_1, fj_2
    %fnmsub = riscv.fnmsub.s %f0, %f1, %f2 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fnmsub.s fj_3, fj_0, fj_1, fj_2
    %fnmadd = riscv.fnmadd.s %f0, %f1, %f2 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fnmadd.s fj_3, fj_0, fj_1, fj_2
    %fadd_s = riscv.fadd.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fadd.s fj_3, fj_0, fj_1
    %fsub_s = riscv.fsub.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fsub.s fj_3, fj_0, fj_1
    %fmul_s = riscv.fmul.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmul.s fj_3, fj_0, fj_1
    %fdiv_s = riscv.fdiv.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fdiv.s fj_3, fj_0, fj_1
    %fsqrt = riscv.fsqrt.s %f0 : (!riscv.freg<fj_0>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fsqrt.s fj_3, fj_0
    %fsgnj = riscv.fsgnj.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fsgnj.s fj_3, fj_0, fj_1
    %fsgnjn = riscv.fsgnjn.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fsgnjn.s fj_3, fj_0, fj_1
    %fsgnjx = riscv.fsgnjx.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fsgnjx.s fj_3, fj_0, fj_1
    %fmin = riscv.fmin.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmin.s fj_3, fj_0, fj_1
    %fmax = riscv.fmax.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmax.s fj_3, fj_0, fj_1
    %fcvtws = riscv.fcvt.w.s %f0 : (!riscv.freg<fj_0>) -> !riscv.reg<j_3>
    // CHECK-NEXT: fcvt.w.s j_3, fj_0
    %fcvtwus = riscv.fcvt.wu.s %f0 : (!riscv.freg<fj_0>) -> !riscv.reg<j_3>
    // CHECK-NEXT: fcvt.wu.s j_3, fj_0
    %fmvxw = riscv.fmv.x.w %f0 : (!riscv.freg<fj_0>) -> !riscv.reg<j_3>
    // CHECK-NEXT: fmv.x.w j_3, fj_0
    %feq = riscv.feq.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.reg<j_3>
    // CHECK-NEXT: feq.s j_3, fj_0, fj_1
    %flt = riscv.flt.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.reg<j_3>
    // CHECK-NEXT: flt.s j_3, fj_0, fj_1
    %fle = riscv.fle.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.reg<j_3>
    // CHECK-NEXT: fle.s j_3, fj_0, fj_1
    %fclass = riscv.fclass.s %f0 : (!riscv.freg<fj_0>) -> !riscv.reg<j_3>
    // CHECK-NEXT: fclass.s j_3, fj_0
    %fcvtsw = riscv.fcvt.s.w %0 : (!riscv.reg<zero>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fcvt.s.w fj_3, zero
    %fcvtswu = riscv.fcvt.s.wu %0 : (!riscv.reg<zero>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fcvt.s.wu fj_3, zero
    %fmvwx = riscv.fmv.w.x %0 : (!riscv.reg<zero>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmv.w.x fj_3, zero
    %flw = riscv.flw %0, 1 : (!riscv.reg<zero>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: flw fj_3, 1(zero)
    riscv.fsw %0, %f0, 1  : (!riscv.reg<zero>, !riscv.freg<fj_0>) -> ()
    // CHECK-NEXT: fsw fj_0, 1(zero)

    // RV32F: 9 "D" Standard Extension for Double-Precision Floating-Point, Version 2.0

    %fld = riscv.fld %0, 1 : (!riscv.reg<zero>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fld fj_3, 1(zero)

    %min_val = riscv.fld %0, "hello" : (!riscv.reg<zero>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fld fj_3, hello, zero

    riscv.fsd %0, %f0, 1  : (!riscv.reg<zero>, !riscv.freg<fj_0>) -> ()
    // CHECK-NEXT: fsd fj_0, 1(zero)

    %fmadd_d = riscv.fmadd.d %f0, %f1, %f2 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmadd.d fj_3, fj_0, fj_1, fj_2
    %fmsub_d = riscv.fmsub.d %f0, %f1, %f2 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>, !riscv.freg<fj_2>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmsub.d fj_3, fj_0, fj_1, fj_2
    %fadd_d= riscv.fadd.d %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fadd.d fj_3, fj_0, fj_1
    %fsub_d = riscv.fsub.d %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fsub.d fj_3, fj_0, fj_1
    %fmul_d = riscv.fmul.d %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmul.d fj_3, fj_0, fj_1
    %fdiv_d = riscv.fdiv.d %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fdiv.d fj_3, fj_0, fj_1
    %fmin_d = riscv.fmin.d %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmin.d fj_3, fj_0, fj_1
    %fmax_d = riscv.fmax.d %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: fmax.d fj_3, fj_0, fj_1

    %fcvt_d_w = riscv.fcvt.d.w %1 : (!riscv.reg<j_1>) -> !riscv.freg<fj_0>
    // CHECK-NEXT: fcvt.d.w fj_0, j_1
    %fcvt_d_wu = riscv.fcvt.d.wu %1 : (!riscv.reg<j_1>) -> !riscv.freg<fj_0>
    // CHECK-NEXT: fcvt.d.wu fj_0, j_1

    // Vector Ops
    %vfadd_s = riscv.vfadd.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: vfadd.s fj_3, fj_0, fj_1
    %vfmul_s = riscv.vfmul.s %f0, %f1 : (!riscv.freg<fj_0>, !riscv.freg<fj_1>) -> !riscv.freg<fj_3>
    // CHECK-NEXT: vfmul.s fj_3, fj_0, fj_1

    // Terminate block
    riscv_func.return
    // CHECK-NEXT: ret
  }
// External
riscv_func.func @external()
// CHECK-NOT: external
}) : () -> ()
