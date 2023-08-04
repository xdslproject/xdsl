builtin.module {
  riscv.label {"label" = #riscv.label<"main">} ({
    %0 = riscv.get_register : () -> !riscv.reg<>
    %1 = riscv.get_register : () -> !riscv.reg<>
    %addi = riscv.addi %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %slti = riscv.slti %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %sltiu = riscv.sltiu %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %andi = riscv.andi %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %ori = riscv.ori %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %xori = riscv.xori %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %slli = riscv.slli %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %srli = riscv.srli %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %srai = riscv.srai %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %lui = riscv.lui 1 : () -> !riscv.reg<>
    %auipc = riscv.auipc 1 : () -> !riscv.reg<>
    %mv = riscv.mv %0 : (!riscv.reg<>) -> !riscv.reg<>
    %add = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %slt = riscv.slt %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %sltu = riscv.sltu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %and = riscv.and %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %or = riscv.or %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %xor = riscv.xor %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %sll = riscv.sll %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %srl = riscv.srl %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %sub = riscv.sub %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %sra = riscv.sra %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    riscv.nop : () -> ()
    riscv.jal 1 : () -> ()
    riscv.jal 1, !riscv.reg<> : () -> ()
    riscv.jal "label" : () -> ()
    riscv.j 1 : () -> ()
    riscv.j "label" : () -> ()
    riscv.jalr %0, 1 : (!riscv.reg<>) -> ()
    riscv.jalr %0, 1, !riscv.reg<> : (!riscv.reg<>) -> ()
    riscv.jalr %0, "label" : (!riscv.reg<>) -> ()
    riscv.ret : () -> ()
  ^0(%2 : !riscv.reg<>, %3 : !riscv.reg<>):
    riscv.beq %0, %1 {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    riscv.bne %0, %1 {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    riscv.blt %0, %1 {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    riscv.bge %0, %1 {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    riscv.bltu %0, %1 {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    riscv.bgeu %0, %1 {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    %lb = riscv.lb %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %lbu = riscv.lbu %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %lh = riscv.lh %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %lhu = riscv.lhu %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    %lw = riscv.lw %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
    riscv.sb %0, %1 {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    riscv.sh %0, %1 {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    riscv.sw %0, %1 {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
    %csrrw_rw = riscv.csrrw %0 {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
    %csrrw_w = riscv.csrrw %0 {"csr" = 1024 : i32, "writeonly"} : (!riscv.reg<>) -> !riscv.reg<>
    %csrrs_rw = riscv.csrrs %0 {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
    %csrrs_r = riscv.csrrs %0 {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<>) -> !riscv.reg<>
    %csrrc_rw = riscv.csrrc %0 {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
    %csrrc_r = riscv.csrrc %0 {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<>) -> !riscv.reg<>
    %csrrsi_rw = riscv.csrrsi {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<>
    %csrrsi_r = riscv.csrrsi {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<>
    %csrrci_rw = riscv.csrrci {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<>
    %csrrci_r = riscv.csrrci {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<>
    %csrrwi_rw = riscv.csrrwi {"csr" = 1024 : i32, "immediate" = 1 : i32} : () -> !riscv.reg<>
    %csrrwi_w = riscv.csrrwi {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32} : () -> !riscv.reg<>
    riscv.wfi : () -> ()
    %mul = riscv.mul %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %mulh = riscv.mulh %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %mulhsu = riscv.mulhsu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %mulhu = riscv.mulhu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %div = riscv.div %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %divu = riscv.divu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %rem = riscv.rem %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %remu = riscv.remu %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %li = riscv.li 1 : () -> !riscv.reg<>
    riscv.ecall : () -> ()
    riscv.ebreak : () -> ()
    riscv.directive {"directive" = ".align", "value" = "2"} : () -> ()
    riscv.assembly_section ".text" attributes {"foo" = i32} {
      %nested_li = riscv.li 1 : () -> !riscv.reg<>
    }
    riscv.assembly_section ".text" {
      %nested_li_1 = riscv.li 1 : () -> !riscv.reg<>
    }
    %custom0, %custom1 = riscv.custom_assembly_instruction %0, %1 {"instruction_name" = "hello"} : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>, !riscv.reg<>)
    riscv.scfgw %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> ()
    %f0 = riscv.get_float_register : () -> !riscv.freg<>
    %f1 = riscv.get_float_register : () -> !riscv.freg<>
    %f2 = riscv.get_float_register : () -> !riscv.freg<>
    %fmadd_s = riscv.fmadd.s %f0, %f1, %f2 : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fmsub_s = riscv.fmsub.s %f0, %f1, %f2 : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fnmsub_s = riscv.fnmsub.s %f0, %f1, %f2 : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fnmadd_s = riscv.fnmadd.s %f0, %f1, %f2 : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fadd_s = riscv.fadd.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fsub_s = riscv.fsub.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fmul_s = riscv.fmul.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fdiv_s = riscv.fdiv.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fsqrt_s = riscv.fsqrt.s %f0 : (!riscv.freg<>) -> !riscv.freg<>
    %fsgnj_s = riscv.fsgnj.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fsgnjn_s = riscv.fsgnjn.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fsgnjx_s = riscv.fsgnjx.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fmin_s = riscv.fmin.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fmax_s = riscv.fmax.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %fcvt_w_s = riscv.fcvt.w.s %f0 : (!riscv.freg<>) -> !riscv.reg<>
    %fcvt_wu_s = riscv.fcvt.wu.s %f0 : (!riscv.freg<>) -> !riscv.reg<>
    %fmv_x_w = riscv.fmv.x.w %f0 : (!riscv.freg<>) -> !riscv.reg<>
    %feq_s = riscv.feq.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    %flt_s = riscv.flt.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    %fle_s = riscv.fle.s %f0, %f1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    %fclass_s = riscv.fclass.s %f0 : (!riscv.freg<>) -> !riscv.reg<>
    %fcvt_s_w = riscv.fcvt.s.w %0 : (!riscv.reg<>) -> !riscv.freg<>
    %fcvt_s_wu = riscv.fcvt.s.wu %0 : (!riscv.reg<>) -> !riscv.freg<>
    %fmv_w_x = riscv.fmv.w.x %0 : (!riscv.reg<>) -> !riscv.freg<>
    %flw = riscv.flw %0 {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.freg<>
    riscv.fsw %0, %f0 {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.freg<>) -> ()
    riscv.ret : () -> ()
  }) : () -> ()
}

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "riscv.label"() ({
// CHECK-NEXT:     %0 = "riscv.get_register"() : () -> !riscv.reg<>
// CHECK-NEXT:     %1 = "riscv.get_register"() : () -> !riscv.reg<>
// CHECK-NEXT:     %addi = "riscv.addi"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %slti = "riscv.slti"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %sltiu = "riscv.sltiu"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %andi = "riscv.andi"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %ori = "riscv.ori"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %xori = "riscv.xori"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %slli = "riscv.slli"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %srli = "riscv.srli"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %srai = "riscv.srai"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %lui = "riscv.lui"() {"immediate" = 1 : ui20} : () -> !riscv.reg<>
// CHECK-NEXT:     %auipc = "riscv.auipc"() {"immediate" = 1 : ui20} : () -> !riscv.reg<>
// CHECK-NEXT:     %mv = "riscv.mv"(%0) : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %add = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %slt = "riscv.slt"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %sltu = "riscv.sltu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %and = "riscv.and"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %or = "riscv.or"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %xor = "riscv.xor"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %sll = "riscv.sll"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %srl = "riscv.srl"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %sub = "riscv.sub"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %sra = "riscv.sra"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     "riscv.nop"() : () -> ()
// CHECK-NEXT:     "riscv.jal"() {"immediate" = 1 : si20} : () -> ()
// CHECK-NEXT:     "riscv.jal"() {"immediate" = 1 : si20, "rd" = !riscv.reg<>} : () -> ()
// CHECK-NEXT:     "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()
// CHECK-NEXT:     "riscv.j"() {"immediate" = 1 : si20} : () -> ()
// CHECK-NEXT:     "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> ()
// CHECK-NEXT:     "riscv.jalr"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.jalr"(%0) {"immediate" = 1 : si12, "rd" = !riscv.reg<>} : (!riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.ret"() : () -> ()
// CHECK-NEXT:   ^0(%2 : !riscv.reg<>, %3 : !riscv.reg<>):
// CHECK-NEXT:     "riscv.beq"(%0, %1) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.bne"(%0, %1) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.blt"(%0, %1) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.bge"(%0, %1) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.bltu"(%0, %1) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.bgeu"(%0, %1) {"offset" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     %lb = "riscv.lb"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %lbu = "riscv.lbu"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %lh = "riscv.lh"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %lhu = "riscv.lhu"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %lw = "riscv.lw"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     "riscv.sb"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.sh"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     "riscv.sw"(%0, %1) {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     %csrrw_rw = "riscv.csrrw"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %csrrw_w = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %csrrs_rw = "riscv.csrrs"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %csrrc_rw = "riscv.csrrc"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:     %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:     %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:     %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:     %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:     %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:     "riscv.wfi"() : () -> ()
// CHECK-NEXT:     %mul = "riscv.mul"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %mulh = "riscv.mulh"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %mulhsu = "riscv.mulhsu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %mulhu = "riscv.mulhu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %div = "riscv.div"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %divu = "riscv.divu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %rem = "riscv.rem"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %remu = "riscv.remu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:     %li = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
// CHECK-NEXT:     "riscv.ecall"() : () -> ()
// CHECK-NEXT:     "riscv.ebreak"() : () -> ()
// CHECK-NEXT:     "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
// CHECK-NEXT:     "riscv.assembly_section"() ({
// CHECK-NEXT:       %nested_li = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
// CHECK-NEXT:     }) {"directive" = ".text", "foo" = i32} : () -> ()
// CHECK-NEXT:     "riscv.assembly_section"() ({
// CHECK-NEXT:       %nested_li_1 = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
// CHECK-NEXT:     }) {"directive" = ".text"} : () -> ()
// CHECK-NEXT:     %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:     "riscv.scfgw"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:     %f0 = "riscv.get_float_register"() : () -> !riscv.freg<>
// CHECK-NEXT:     %f1 = "riscv.get_float_register"() : () -> !riscv.freg<>
// CHECK-NEXT:     %f2 = "riscv.get_float_register"() : () -> !riscv.freg<>
// CHECK-NEXT:     %fmadd_s = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fmsub_s = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fnmsub_s = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fnmadd_s = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fadd_s = "riscv.fadd.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fsub_s = "riscv.fsub.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fmul_s = "riscv.fmul.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fdiv_s = "riscv.fdiv.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fsqrt_s = "riscv.fsqrt.s"(%f0) : (!riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fsgnj_s = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fsgnjn_s = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fsgnjx_s = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fmin_s = "riscv.fmin.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fmax_s = "riscv.fmax.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fcvt_w_s = "riscv.fcvt.w.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
// CHECK-NEXT:     %fcvt_wu_s = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
// CHECK-NEXT:     %fmv_x_w = "riscv.fmv.x.w"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
// CHECK-NEXT:     %feq_s = "riscv.feq.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
// CHECK-NEXT:     %flt_s = "riscv.flt.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
// CHECK-NEXT:     %fle_s = "riscv.fle.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
// CHECK-NEXT:     %fclass_s = "riscv.fclass.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
// CHECK-NEXT:     %fcvt_s_w = "riscv.fcvt.s.w"(%0) : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fcvt_s_wu = "riscv.fcvt.s.wu"(%0) : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:     %fmv_w_x = "riscv.fmv.w.x"(%0) : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:     %flw = "riscv.flw"(%0) {"immediate" = 1 : i32} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:     "riscv.fsw"(%0, %f0) {"immediate" = 1 : i32} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-NEXT:     "riscv.ret"() : () -> ()
// CHECK-NEXT:   }) {"label" = #riscv.label<"main">} : () -> ()
// CHECK-NEXT: }) : () -> ()
