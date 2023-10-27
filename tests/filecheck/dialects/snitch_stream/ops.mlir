// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%0 = riscv.get_register : () -> !riscv.reg<>
%1 = riscv.get_register : () -> !riscv.reg<>
// RV32I/RV64I: 2.4 Integer Computational Instructions

// Integer Register-Immediate Instructions
%addi = riscv.addi %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK: %{{.*}} = riscv.addi %{{.*}}, 1 : (!riscv.reg<>) -> !riscv.reg<>
%slti = riscv.slti %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.slti %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
%sltiu = riscv.sltiu %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.sltiu %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
%andi = riscv.andi %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.andi %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
%ori = riscv.ori %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.ori %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
%xori = riscv.xori %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.xori %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
%slli = riscv.slli %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.slli %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
%srli = riscv.srli %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.srli %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
%srai = riscv.srai %0, 1: (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.srai %0, 1 : (!riscv.reg<>) -> !riscv.reg<>
%lui = riscv.lui 1 : () -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.lui 1 : () -> !riscv.reg<>
%auipc = riscv.auipc 1 : () -> !riscv.reg<>
// CHECK-NEXT: %{{.*}} = riscv.auipc 1 : () -> !riscv.reg<>
%mv = riscv.mv %0 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK: %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<>

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:     %0 = "riscv.get_register"() : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %1 = "riscv.get_register"() : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %addi = "riscv.addi"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %slti = "riscv.slti"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %sltiu = "riscv.sltiu"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %andi = "riscv.andi"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %ori = "riscv.ori"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %xori = "riscv.xori"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %slli = "riscv.slli"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %srli = "riscv.srli"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %srai = "riscv.srai"(%0) {"immediate" = 1 : ui5} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %lui = "riscv.lui"() {"immediate" = 1 : ui20} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %auipc = "riscv.auipc"() {"immediate" = 1 : ui20} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %mv = "riscv.mv"(%0) : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %add = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %slt = "riscv.slt"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %sltu = "riscv.sltu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %and = "riscv.and"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %or = "riscv.or"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %xor = "riscv.xor"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %sll = "riscv.sll"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %srl = "riscv.srl"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %sub = "riscv.sub"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %sra = "riscv.sra"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     "riscv.nop"() : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.jal"() {"immediate" = 1 : si20} : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.jal"() {"immediate" = 1 : si20, "rd" = !riscv.reg<>} : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.j"() {"immediate" = 1 : si20} : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.jalr"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.jalr"(%0) {"immediate" = 1 : si12, "rd" = !riscv.reg<>} : (!riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.ret"() : () -> ()
// CHECK-GENERIC-NEXT:   ^0(%2 : !riscv.reg<>, %3 : !riscv.reg<>):
// CHECK-GENERIC-NEXT:     "riscv.beq"(%0, %1) {"offset" = 1 : i12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.bne"(%0, %1) {"offset" = 1 : i12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.blt"(%0, %1) {"offset" = 1 : i12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.bge"(%0, %1) {"offset" = 1 : i12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.bltu"(%0, %1) {"offset" = 1 : i12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.bgeu"(%0, %1) {"offset" = 1 : i12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     %lb = "riscv.lb"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %lbu = "riscv.lbu"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %lh = "riscv.lh"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %lhu = "riscv.lhu"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %lw = "riscv.lw"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     "riscv.sb"(%0, %1) {"immediate" = 1 : si12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.sh"(%0, %1) {"immediate" = 1 : si12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     "riscv.sw"(%0, %1) {"immediate" = 1 : si12} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-GENERIC-NEXT:     %csrrw_rw = "riscv.csrrw"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrw_w = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrs_rw = "riscv.csrrs"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrc_rw = "riscv.csrrc"(%0) {"csr" = 1024 : i32} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32, "writeonly"} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     "riscv.wfi"() : () -> ()
// CHECK-GENERIC-NEXT:     %mul = "riscv.mul"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %mulh = "riscv.mulh"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %mulhsu = "riscv.mulhsu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %mulhu = "riscv.mulhu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %div = "riscv.div"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %divu = "riscv.divu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %rem = "riscv.rem"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %remu = "riscv.remu"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %li = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     "riscv.ecall"() : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.ebreak"() : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.directive"() {"directive" = ".bss"} : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.assembly_section"() ({
// CHECK-GENERIC-NEXT:       %nested_li = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     }) {"directive" = ".text", "foo" = i32} : () -> ()
// CHECK-GENERIC-NEXT:     "riscv.assembly_section"() ({
// CHECK-GENERIC-NEXT:       %nested_li_1 = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     }) {"directive" = ".text"} : () -> ()
// CHECK-GENERIC-NEXT:     %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.reg<>, !riscv.reg<>) -> (!riscv.reg<>, !riscv.reg<>)
// CHECK-GENERIC-NEXT:     %f0 = "riscv.get_float_register"() : () -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %f1 = "riscv.get_float_register"() : () -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %f2 = "riscv.get_float_register"() : () -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fmv = "riscv.fmv.s"(%f0) : (!riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fmadd_s = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fmsub_s = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fnmsub_s = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fnmadd_s = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fadd_s = "riscv.fadd.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fsub_s = "riscv.fsub.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fmul_s = "riscv.fmul.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fdiv_s = "riscv.fdiv.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fsqrt_s = "riscv.fsqrt.s"(%f0) : (!riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fsgnj_s = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fsgnjn_s = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fsgnjx_s = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fmin_s = "riscv.fmin.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fmax_s = "riscv.fmax.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fcvt_w_s = "riscv.fcvt.w.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %fcvt_wu_s = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %fmv_x_w = "riscv.fmv.x.w"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %feq_s = "riscv.feq.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %flt_s = "riscv.flt.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %fle_s = "riscv.fle.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %fclass_s = "riscv.fclass.s"(%f0) : (!riscv.freg<>) -> !riscv.reg<>
// CHECK-GENERIC-NEXT:     %fcvt_s_w = "riscv.fcvt.s.w"(%0) : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fcvt_s_wu = "riscv.fcvt.s.wu"(%0) : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %fmv_w_x = "riscv.fmv.w.x"(%0) : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %flw = "riscv.flw"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     "riscv.fsw"(%0, %f0) {"immediate" = 1 : si12} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-GENERIC-NEXT:     %fld = "riscv.fld"(%0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     "riscv.fsd"(%0, %f0) {"immediate" = 1 : si12} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-GENERIC-NEXT:     %vfadd_s = "riscv.vfadd.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     %vfmul_s = "riscv.vfmul.s"(%f0, %f1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-GENERIC-NEXT:     "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) {"sym_name" = "main", "function_type" = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
