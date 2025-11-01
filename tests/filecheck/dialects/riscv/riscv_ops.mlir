// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

"builtin.module"() ({
  riscv_func.func @main() {
    %0 = riscv.get_register : !riscv.reg
    %1 = riscv.get_register : !riscv.reg
    // RV32I/RV64I: 2.4 Integer Computational Instructions

    // Integer Register-Immediate Instructions
    %addi = riscv.addi %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK: %{{.*}} = riscv.addi %{{.*}}, 1 : (!riscv.reg) -> !riscv.reg
    %slti = riscv.slti %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.slti %0, 1 : (!riscv.reg) -> !riscv.reg
    %sltiu = riscv.sltiu %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltiu %0, 1 : (!riscv.reg) -> !riscv.reg
    %andi = riscv.andi %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.andi %0, 1 : (!riscv.reg) -> !riscv.reg
    %ori = riscv.ori %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.ori %0, 1 : (!riscv.reg) -> !riscv.reg
    %xori = riscv.xori %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %0, 1 : (!riscv.reg) -> !riscv.reg
    %slli = riscv.slli %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.slli %0, 1 : (!riscv.reg) -> !riscv.reg
    %srli = riscv.srli %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.srli %0, 1 : (!riscv.reg) -> !riscv.reg
    %srai = riscv.srai %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.srai %0, 1 : (!riscv.reg) -> !riscv.reg
    %lui = riscv.lui 1 : () -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.lui 1 : () -> !riscv.reg
    %auipc = riscv.auipc 1 : () -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.auipc 1 : () -> !riscv.reg
    %mv = riscv.mv %0 : (!riscv.reg) -> !riscv.reg
    // CHECK: %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg
    %seqz = riscv.seqz %0 : (!riscv.reg) -> !riscv.reg
    // CHECK: %{{.*}} = riscv.seqz %{{.*}} : (!riscv.reg) -> !riscv.reg
    %snez = riscv.snez %0 : (!riscv.reg) -> !riscv.reg
    // CHECK: %{{.*}} = riscv.snez %{{.*}} : (!riscv.reg) -> !riscv.reg
    %zextb = riscv.zext.b %0 : (!riscv.reg) -> !riscv.reg
    // CHECK: %{{.*}} = riscv.zext.b %{{.*}} : (!riscv.reg) -> !riscv.reg
    %zextw = riscv.zext.w %0 : (!riscv.reg) -> !riscv.reg
    // CHECK: %{{.*}} = riscv.zext.w %{{.*}} : (!riscv.reg) -> !riscv.reg
    %sextw = riscv.sext.w %0 : (!riscv.reg) -> !riscv.reg
    // CHECK: %{{.*}} = riscv.sext.w %{{.*}} : (!riscv.reg) -> !riscv.reg
    %srliw = riscv.srliw %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.srliw %0, 1 : (!riscv.reg) -> !riscv.reg
    %sraiw = riscv.sraiw %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sraiw %0, 1 : (!riscv.reg) -> !riscv.reg

    // Integer Register-Register Operations
    %add = riscv.add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %slt = riscv.slt %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.slt %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sltu = riscv.sltu %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltu %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %and = riscv.and %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %or = riscv.or %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %xor = riscv.xor %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xor %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sll = riscv.sll %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sll %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %srl = riscv.srl %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.srl %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sub = riscv.sub %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sub %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sra = riscv.sra %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sra %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %addw = riscv.addw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.addw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %subw = riscv.subw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.subw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sllw = riscv.sllw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sllw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %srlw = riscv.srlw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.srlw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sraw = riscv.sraw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sraw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    riscv.nop
    // CHECK-NEXT: riscv.nop

    // RV32I/RV64I: 2.5 Control Transfer Instructions

    // Unconditional Branch Instructions
    riscv.jal 1
    // CHECK-NEXT: riscv.jal 1
    riscv.jal 1, !riscv.reg
    // CHECK-NEXT: riscv.jal 1, !riscv.reg
    riscv.jal "label"
    // CHECK-NEXT: riscv.jal "label"

    riscv.j 1
    // CHECK-NEXT: riscv.j 1
    riscv.j "label"
    // CHECK-NEXT: riscv.j "label"

    riscv.jalr %0, 1: (!riscv.reg) -> ()
    // CHECK-NEXT: riscv.jalr %0, 1 : (!riscv.reg) -> ()
    riscv.jalr %0, 1, !riscv.reg : (!riscv.reg) -> ()
    // CHECK-NEXT: riscv.jalr %0, 1, !riscv.reg : (!riscv.reg) -> ()
    riscv.jalr %0, "label" : (!riscv.reg) -> ()
    // CHECK-NEXT: riscv.jalr %0, "label" : (!riscv.reg) -> ()

    riscv.ret
    // CHECK-NEXT: riscv.ret
  ^bb0(%2 : !riscv.reg, %3 : !riscv.reg):
  // CHECK-NEXT: ^bb0(%2 : !riscv.reg, %3 : !riscv.reg):

    // Conditional Branch Instructions
    riscv.beq %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.beq %{{.*}}, %{{.*}}, 1 : (!riscv.reg, !riscv.reg) -> ()
    riscv.bne %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.bne %{{.*}}, %{{.*}}, 1 : (!riscv.reg, !riscv.reg) -> ()
    riscv.blt %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.blt %{{.*}}, %{{.*}}, 1 : (!riscv.reg, !riscv.reg) -> ()
    riscv.bge %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.bge %{{.*}}, %{{.*}}, 1 : (!riscv.reg, !riscv.reg) -> ()
    riscv.bltu %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.bltu %{{.*}}, %{{.*}}, 1 : (!riscv.reg, !riscv.reg) -> ()
    riscv.bgeu %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.bgeu %{{.*}}, %{{.*}}, 1 : (!riscv.reg, !riscv.reg) -> ()

    // RV32I/RV64I: 2.6 Load and Store Instructions

    %lb = riscv.lb %0, 1 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.lb %0, 1 : (!riscv.reg) -> !riscv.reg
    %lbu = riscv.lbu %0, 1: (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.lbu %0, 1 : (!riscv.reg) -> !riscv.reg
    %lh = riscv.lh %0, 1 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.lh %0, 1 : (!riscv.reg) -> !riscv.reg
    %lhu = riscv.lhu %0, 1 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.lhu %0, 1 : (!riscv.reg) -> !riscv.reg
    %lw = riscv.lw %0, 1 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.lw %0, 1 : (!riscv.reg) -> !riscv.reg
    riscv.sb %0, %1, 1: (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.sb %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    riscv.sh %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.sh %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    riscv.sw %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: riscv.sw %0, %1, 1 : (!riscv.reg, !riscv.reg) -> ()

    // RV32I/RV64I: 2.8 Control and Status Register Instructions

    %csrrw_rw = riscv.csrrw %0 1024 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrw %0, 1024 : (!riscv.reg) -> !riscv.reg
    %csrrw_w = riscv.csrrw %0 1024, "w" : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrw %0, 1024, "w" : (!riscv.reg) -> !riscv.reg
    %csrrs_rw = riscv.csrrs %0, 1024 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrs %0, 1024 : (!riscv.reg) -> !riscv.reg
    %csrrs_r = riscv.csrrs %0, 1024, "r" : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrs %0, 1024, "r" : (!riscv.reg) -> !riscv.reg
    %csrrc_rw = riscv.csrrc %0, 1024 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrc %0, 1024 : (!riscv.reg) -> !riscv.reg
    %csrrc_r = riscv.csrrc %0, 1024, "r" : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrc %0, 1024, "r" : (!riscv.reg) -> !riscv.reg
    %csrrsi_rw = riscv.csrrsi 1024, 8 : () -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrsi 1024, 8 : () -> !riscv.reg
    %csrrsi_r = riscv.csrrsi 1024, 0 : () -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrsi 1024, 0 : () -> !riscv.reg
    %csrrci_rw = riscv.csrrci 1024, 8 : () -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrci 1024, 8 : () -> !riscv.reg
    %csrrci_r = riscv.csrrci 1024, 0 : () -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrci 1024, 0 : () -> !riscv.reg
    %csrrwi_rw = riscv.csrrwi 1024, 1 : () -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrwi 1024, 1 : () -> !riscv.reg
    %csrrwi_w = riscv.csrrwi 1024, 1, "w" : () -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.csrrwi 1024, 1, "w" : () -> !riscv.reg

    // Machine Mode Privileged Instructions
    riscv.wfi
    // CHECK-NEXT: riscv.wfi


    // RV32M/RV64M: 7 “M” Standard Extension for Integer Multiplication and Division

    // Multiplication Operations
    %mul = riscv.mul %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %mulh = riscv.mulh %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.mulh %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %mulhsu = riscv.mulhsu %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.mulhsu %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %mulhu = riscv.mulhu %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.mulhu %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %mulw = riscv.mulw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.mulw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg


    // Division Operations
    %div = riscv.div %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.div %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %divu = riscv.divu %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.divu %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %divw = riscv.divw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.divw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %divuw = riscv.divuw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.divuw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %rem = riscv.rem %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.rem %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %remu = riscv.remu %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.remu %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %remw = riscv.remw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.remw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %remuw = riscv.remuw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.remuw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    
    // RV32B/RV64B:  “B” Standard Extension for Bit Manipulation

    %rol = riscv.rol %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.rol %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %ror = riscv.ror %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.ror %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %rori = riscv.rori %0, 1 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.rori %{{.*}}, 1 : (!riscv.reg) -> !riscv.reg
    %roriw = riscv.roriw %0, 1 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.roriw %{{.*}}, 1 : (!riscv.reg) -> !riscv.reg
    %rolw = riscv.rolw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.rolw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %rorw = riscv.rorw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.rorw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %bclr = riscv.bclr %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.bclr %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %bext = riscv.bext %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.bext %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %bset = riscv.bset %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.bset %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %binv = riscv.binv %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.binv %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %adduw = riscv.add.uw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.add.uw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sh1add = riscv.sh1add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sh1add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sh1adduw = riscv.sh1add.uw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sh1add.uw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sh2add = riscv.sh2add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sh2add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sh2adduw = riscv.sh2add.uw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sh2add.uw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sh3add = riscv.sh3add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sh3add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sh3adduw = riscv.sh3add.uw %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sh3add.uw %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %sextb = riscv.sext.b %0 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sext.b %{{.*}} : (!riscv.reg) -> !riscv.reg
    %sexth = riscv.sext.h %0 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sext.h %{{.*}} : (!riscv.reg) -> !riscv.reg
    %zexth = riscv.zext.h %0 : (!riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.zext.h %{{.*}} : (!riscv.reg) -> !riscv.reg
    %andn = riscv.andn %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.andn %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %xnor = riscv.xnor %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xnor %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %orn = riscv.orn %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.orn %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %max = riscv.max %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.max %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %maxu = riscv.maxu %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.maxu %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %min = riscv.min %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.min %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %minu = riscv.minu %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.minu %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg


    // RV32/RV64 Zicond extension for conditional operations 

    %czeroeqzop = riscv.czero.eqz %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.czero.eqz %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %czeronezop = riscv.czero.nez %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.czero.nez %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
    
    // Assembler pseudo-instructions

    %li = riscv.li 1 : !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.li 1 : !riscv.reg
    // Environment Call and Breakpoints
    riscv.ecall
    // CHECK-NEXT: riscv.ecall
    riscv.ebreak
    // CHECK-NEXT: riscv.ebreak
    riscv.directive ".bss"
    // CHECK-NEXT: riscv.directive ".bss"
    riscv.directive ".align" "2"
    // CHECK-NEXT: riscv.directive ".align" "2"
    riscv.assembly_section ".text" attributes {"foo" = i32} {
      %nested_li = riscv.li 1 : !riscv.reg
    }
    // CHECK-NEXT:  riscv.assembly_section ".text" attributes {foo = i32} {
    // CHECK-NEXT:    %{{.*}} = riscv.li 1 : !riscv.reg
    // CHECK-NEXT:  }

    riscv.assembly_section ".text" {
      %nested_li = riscv.li 1 : !riscv.reg
    }
    // CHECK-NEXT:  riscv.assembly_section ".text" {
    // CHECK-NEXT:    %{{.*}} = riscv.li 1 : !riscv.reg
    // CHECK-NEXT:  }

    // Custom instruction
    %custom0, %custom1 = riscv.custom_assembly_instruction %0, %1 {"instruction_name" = "hello"} : (!riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg)
    // CHECK-NEXT:   %custom0, %custom1 = riscv.custom_assembly_instruction %0, %1 {instruction_name = "hello"} : (!riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg)

    // RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0
    %f0 = riscv.get_float_register : !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.get_float_register : !riscv.freg
    %f1 = riscv.get_float_register : !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.get_float_register : !riscv.freg
    %f2 = riscv.get_float_register : !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.get_float_register : !riscv.freg

    %fmv = riscv.fmv.s %f0 : (!riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg

    %fmadd_s = riscv.fmadd.s %f0, %f1, %f2 : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmadd.s %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    %fmsub_s = riscv.fmsub.s %f0, %f1, %f2 : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmsub.s %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    %fnmsub_s = riscv.fnmsub.s %f0, %f1, %f2 : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fnmsub.s %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    %fnmadd_s = riscv.fnmadd.s %f0, %f1, %f2 : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fnmadd.s %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg

    %fadd_s = riscv.fadd.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fsub_s = riscv.fsub.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fmul_s = riscv.fmul.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fdiv_s = riscv.fdiv.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fadd_s_fm = riscv.fadd.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fsub_s_fm = riscv.fsub.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fmul_s_fm = riscv.fmul.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fdiv_s_fm = riscv.fdiv.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fsqrt_s = riscv.fsqrt.s %f0 : (!riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsqrt.s %{{.*}} : (!riscv.freg) -> !riscv.freg

    %fsgnj_s = riscv.fsgnj.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsgnj.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fsgnjn_s = riscv.fsgnjn.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsgnjn.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fsgnjx_s = riscv.fsgnjx.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsgnjx.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fmin_s = riscv.fmin.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fmax_s = riscv.fmax.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fmin_s_fm = riscv.fmin.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fmax_s_fm = riscv.fmax.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fcvt_w_s = riscv.fcvt.w.s %f0 : (!riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.w.s %{{.*}} : (!riscv.freg) -> !riscv.reg
    %fcvt_wu_s = riscv.fcvt.wu.s %f0 : (!riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.wu.s %{{.*}} : (!riscv.freg) -> !riscv.reg
    %fmv_x_w = riscv.fmv.x.w %f0 : (!riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fmv.x.w %{{.*}} : (!riscv.freg) -> !riscv.reg

    %feq_s = riscv.feq.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %flt_s = riscv.flt.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %fle_s = riscv.fle.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %feq_s_fm = riscv.feq.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %flt_s_fm = riscv.flt.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %fle_s_fm = riscv.fle.s %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %fclass_s = riscv.fclass.s %f0 : (!riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fclass.s %{{.*}} : (!riscv.freg) -> !riscv.reg
    %fcvt_s_w = riscv.fcvt.s.w %0 : (!riscv.reg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.s.w %{{.*}} : (!riscv.reg) -> !riscv.freg
    %fcvt_s_wu = riscv.fcvt.s.wu %0 : (!riscv.reg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.s.wu %{{.*}} : (!riscv.reg) -> !riscv.freg
    %fmv_w_x = riscv.fmv.w.x %0 : (!riscv.reg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmv.w.x %{{.*}} : (!riscv.reg) -> !riscv.freg

    %flw = riscv.flw %0, 1 : (!riscv.reg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.flw %{{.*}}, 1 : (!riscv.reg) -> !riscv.freg
    riscv.fsw %0, %f0, 1 : (!riscv.reg, !riscv.freg) -> ()
    // CHECK-NEXT: riscv.fsw %{{.*}}, %{{.*}}, 1 : (!riscv.reg, !riscv.freg) -> ()

    // Vector Ops
    %fld = riscv.fld %0, 1 : (!riscv.reg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fld %{{.*}}, 1 : (!riscv.reg) -> !riscv.freg
    riscv.fsd %0, %f0, 1 : (!riscv.reg, !riscv.freg) -> ()
    // CHECK-NEXT: riscv.fsd %{{.*}}, %{{.*}}, 1 : (!riscv.reg, !riscv.freg) -> ()

    %fmv_d = riscv.fmv.d %f0 : (!riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg) -> !riscv.freg

    %vfadd_s = riscv.vfadd.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.vfadd.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %vfmul_s = riscv.vfmul.s %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.vfmul.s %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg

    // RV32F: 9 “D” Standard Extension for Double-Precision Floating-Point, Version 2.0

    %fadd_d = riscv.fadd.d %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.d %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fsub_d = riscv.fsub.d %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.d %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fmul_d = riscv.fmul.d %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.d %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fdiv_d = riscv.fdiv.d %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.d %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fadd_d_fm = riscv.fadd.d %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.d %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fsub_d_fm = riscv.fsub.d %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.d %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fmul_d_fm = riscv.fmul.d %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.d %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fdiv_d_fm = riscv.fdiv.d %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.d %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fmadd_d = riscv.fmadd.d %f0, %f1, %f2 : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmadd.d %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    %fmsub_d = riscv.fmsub.d %f0, %f1, %f2 : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmsub.d %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg

    %fmin_d = riscv.fmin.d %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.d %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fmax_d = riscv.fmax.d %f0, %f1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.d %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fmin_d_fm = riscv.fmin.d %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.d %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %fmax_d_fm = riscv.fmax.d %f0, %f1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.d %{{.*}}, %{{.*}} fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %fcvt_d_w = riscv.fcvt.d.w %0 : (!riscv.reg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.d.w %{{.*}} : (!riscv.reg) -> !riscv.freg
    %fcvt_d_wu = riscv.fcvt.d.wu %0 : (!riscv.reg) -> !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.d.wu %{{.*}} : (!riscv.reg) -> !riscv.freg

    // Terminate block
    riscv_func.return
  }
}) : () -> ()

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:      %0 = "riscv.get_register"() : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %1 = "riscv.get_register"() : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %addi = "riscv.addi"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %slti = "riscv.slti"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sltiu = "riscv.sltiu"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %andi = "riscv.andi"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %ori = "riscv.ori"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %xori = "riscv.xori"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %slli = "riscv.slli"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srli = "riscv.srli"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srai = "riscv.srai"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %lui = "riscv.lui"() {immediate = 1 : i20} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %auipc = "riscv.auipc"() {immediate = 1 : i20} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %mv = "riscv.mv"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %seqz = "riscv.seqz"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %snez = "riscv.snez"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %zextb = "riscv.zext.b"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %zextw = "riscv.zext.w"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sextw = "riscv.sext.w"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srliw = "riscv.srliw"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sraiw = "riscv.sraiw"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %add = "riscv.add"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %slt = "riscv.slt"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sltu = "riscv.sltu"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %and = "riscv.and"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %or = "riscv.or"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %xor = "riscv.xor"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sll = "riscv.sll"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srl = "riscv.srl"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sub = "riscv.sub"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sra = "riscv.sra"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %addw = "riscv.addw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %subw = "riscv.subw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sllw = "riscv.sllw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %srlw = "riscv.srlw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sraw = "riscv.sraw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv.nop"() : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.jal"() {immediate = 1 : si20} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.jal"() {immediate = 1 : si20, rd = !riscv.reg} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.jal"() {immediate = #riscv.label<"label">} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.j"() {immediate = 1 : si20} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.j"() {immediate = #riscv.label<"label">} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.jalr"(%0) {immediate = 1 : si12} : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.jalr"(%0) {immediate = 1 : si12, rd = !riscv.reg} : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.jalr"(%0) {immediate = #riscv.label<"label">} : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.ret"() : () -> ()
// CHECK-GENERIC-NEXT:    ^bb0(%2 : !riscv.reg, %3 : !riscv.reg):
// CHECK-GENERIC-NEXT:      "riscv.beq"(%0, %1) {offset = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.bne"(%0, %1) {offset = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.blt"(%0, %1) {offset = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.bge"(%0, %1) {offset = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.bltu"(%0, %1) {offset = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.bgeu"(%0, %1) {offset = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      %lb = "riscv.lb"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %lbu = "riscv.lbu"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %lh = "riscv.lh"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %lhu = "riscv.lhu"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %lw = "riscv.lw"(%0) {immediate = 1 : si12} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv.sb"(%0, %1) {immediate = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.sh"(%0, %1) {immediate = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      "riscv.sw"(%0, %1) {immediate = 1 : si12} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:      %csrrw_rw = "riscv.csrrw"(%0) {csr = 1024 : i32} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrw_w = "riscv.csrrw"(%0) {csr = 1024 : i32, writeonly} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrs_rw = "riscv.csrrs"(%0) {csr = 1024 : i32} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrs_r = "riscv.csrrs"(%0) {csr = 1024 : i32, readonly} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrc_rw = "riscv.csrrc"(%0) {csr = 1024 : i32} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrc_r = "riscv.csrrc"(%0) {csr = 1024 : i32, readonly} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrsi_rw = "riscv.csrrsi"() {csr = 1024 : i32, immediate = 8 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrsi_r = "riscv.csrrsi"() {csr = 1024 : i32, immediate = 0 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrci_rw = "riscv.csrrci"() {csr = 1024 : i32, immediate = 8 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrci_r = "riscv.csrrci"() {csr = 1024 : i32, immediate = 0 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrwi_rw = "riscv.csrrwi"() {csr = 1024 : i32, immediate = 1 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      %csrrwi_w = "riscv.csrrwi"() {csr = 1024 : i32, immediate = 1 : i32, writeonly} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv.wfi"() : () -> ()
// CHECK-GENERIC-NEXT:      %mul = "riscv.mul"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %mulh = "riscv.mulh"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %mulhsu = "riscv.mulhsu"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %mulhu = "riscv.mulhu"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %mulw = "riscv.mulw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %div = "riscv.div"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %divu = "riscv.divu"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %divw = "riscv.divw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %divuw = "riscv.divuw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %rem = "riscv.rem"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %remu = "riscv.remu"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %remw = "riscv.remw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %remuw = "riscv.remuw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %rol = "riscv.rol"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %ror = "riscv.ror"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %rori = "riscv.rori"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %roriw = "riscv.roriw"(%0) {immediate = 1 : ui5} : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %rolw = "riscv.rolw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %rorw = "riscv.rorw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %bclr = "riscv.bclr"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %bext = "riscv.bext"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %bset = "riscv.bset"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %binv = "riscv.binv"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %adduw = "riscv.add.uw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sh1add = "riscv.sh1add"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sh1adduw = "riscv.sh1add.uw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sh2add = "riscv.sh2add"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sh2adduw = "riscv.sh2add.uw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sh3add = "riscv.sh3add"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sh3adduw = "riscv.sh3add.uw"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sextb = "riscv.sext.b"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %sexth = "riscv.sext.h"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %zexth = "riscv.zext.h"(%0) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %andn = "riscv.andn"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %xnor = "riscv.xnor"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %orn = "riscv.orn"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %max = "riscv.max"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %maxu = "riscv.maxu"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %min = "riscv.min"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %minu = "riscv.minu"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %czeroeqzop = "riscv.czero.eqz"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %czeronezop = "riscv.czero.nez"(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %li = "riscv.li"() {immediate = 1 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv.ecall"() : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.ebreak"() : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.directive"() {directive = ".bss"} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.directive"() {directive = ".align", value = "2"} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.assembly_section"() ({
// CHECK-GENERIC-NEXT:        %nested_li = "riscv.li"() {immediate = 1 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      }) {directive = ".text", foo = i32} : () -> ()
// CHECK-GENERIC-NEXT:      "riscv.assembly_section"() ({
// CHECK-GENERIC-NEXT:        %nested_li = "riscv.li"() {immediate = 1 : i32} : () -> !riscv.reg
// CHECK-GENERIC-NEXT:      }) {directive = ".text"} : () -> ()
// CHECK-GENERIC-NEXT:      %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {instruction_name = "hello"} : (!riscv.reg, !riscv.reg) -> (!riscv.reg, !riscv.reg)
// CHECK-GENERIC-NEXT:      %f0 = "riscv.get_float_register"() : () -> !riscv.freg
// CHECK-GENERIC-NEXT:      %f1 = "riscv.get_float_register"() : () -> !riscv.freg
// CHECK-GENERIC-NEXT:      %f2 = "riscv.get_float_register"() : () -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmv = "riscv.fmv.s"(%f0) : (!riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmadd_s = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmsub_s = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fnmsub_s = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fnmadd_s = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fadd_s = "riscv.fadd.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fsub_s = "riscv.fsub.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmul_s = "riscv.fmul.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fdiv_s = "riscv.fdiv.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fadd_s_fm = "riscv.fadd.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fsub_s_fm = "riscv.fsub.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmul_s_fm = "riscv.fmul.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fdiv_s_fm = "riscv.fdiv.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fsqrt_s = "riscv.fsqrt.s"(%f0) : (!riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fsgnj_s = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fsgnjn_s = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fsgnjx_s = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmin_s = "riscv.fmin.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmax_s = "riscv.fmax.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmin_s_fm = "riscv.fmin.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmax_s_fm = "riscv.fmax.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fcvt_w_s = "riscv.fcvt.w.s"(%f0) : (!riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %fcvt_wu_s = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %fmv_x_w = "riscv.fmv.x.w"(%f0) : (!riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %feq_s = "riscv.feq.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %flt_s = "riscv.flt.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %fle_s = "riscv.fle.s"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %feq_s_fm = "riscv.feq.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %flt_s_fm = "riscv.flt.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %fle_s_fm = "riscv.fle.s"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %fclass_s = "riscv.fclass.s"(%f0) : (!riscv.freg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      %fcvt_s_w = "riscv.fcvt.s.w"(%0) : (!riscv.reg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fcvt_s_wu = "riscv.fcvt.s.wu"(%0) : (!riscv.reg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmv_w_x = "riscv.fmv.w.x"(%0) : (!riscv.reg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %flw = "riscv.flw"(%0) {immediate = 1 : i12} : (!riscv.reg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      "riscv.fsw"(%0, %f0) {immediate = 1 : i12} : (!riscv.reg, !riscv.freg) -> ()
// CHECK-GENERIC-NEXT:      %fld = "riscv.fld"(%0) {immediate = 1 : i12} : (!riscv.reg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      "riscv.fsd"(%0, %f0) {immediate = 1 : i12} : (!riscv.reg, !riscv.freg) -> ()
// CHECK-GENERIC-NEXT:      %fmv_d = "riscv.fmv.d"(%f0) : (!riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %vfadd_s = "riscv.vfadd.s"(%f0, %f1) : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %vfmul_s = "riscv.vfmul.s"(%f0, %f1) : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fadd_d = "riscv.fadd.d"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fsub_d = "riscv.fsub.d"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmul_d = "riscv.fmul.d"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fdiv_d = "riscv.fdiv.d"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fadd_d_fm = "riscv.fadd.d"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fsub_d_fm = "riscv.fsub.d"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmul_d_fm = "riscv.fmul.d"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fdiv_d_fm = "riscv.fdiv.d"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmadd_d = "riscv.fmadd.d"(%f0, %f1, %f2) : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmsub_d = "riscv.fmsub.d"(%f0, %f1, %f2) : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmin_d = "riscv.fmin.d"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmax_d = "riscv.fmax.d"(%f0, %f1) {fastmath = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmin_d_fm = "riscv.fmin.d"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fmax_d_fm = "riscv.fmax.d"(%f0, %f1) {fastmath = #riscv.fastmath<fast>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fcvt_d_w = "riscv.fcvt.d.w"(%0) : (!riscv.reg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      %fcvt_d_wu = "riscv.fcvt.d.wu"(%0) : (!riscv.reg) -> !riscv.freg
// CHECK-GENERIC-NEXT:      "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()
