// RUN: xdsl-opt %s | filecheck %s
"builtin.module"() ({
  %0 = "riscv.get_register"() : () -> !riscv.reg<x$>
  %1 = "riscv.get_register"() : () -> !riscv.reg<x$>
  // CHECK: %{{.*}} = riscv.get_register : -> x$
  // CHECK-NEXT: %{{.*}} = riscv.get_register : -> x$
  // RV32I/RV64I: 2.4 Integer Computational Instructions

  // Integer Register-Immediate Instructions
  %addi = "riscv.addi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %slti = "riscv.slti"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %sltiu = "riscv.sltiu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %andi = "riscv.andi"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %ori = "riscv.ori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %xori = "riscv.xori"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %slli = "riscv.slli"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %srli = "riscv.srli"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %srai = "riscv.srai"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lui = "riscv.lui"() {"immediate" = 1 : i32}: () -> !riscv.reg<x$>
  %auipc = "riscv.auipc"() {"immediate" = 1 : i32}: () -> !riscv.reg<x$>
  %mv = "riscv.mv"(%0) : (!riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %addi = riscv.addi %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %slti = riscv.slti %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %sltiu = riscv.sltiu %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %andi = riscv.andi %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %ori = riscv.ori %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %xori = riscv.xori %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %slli = riscv.slli %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %srli = riscv.srli %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %srai = riscv.srai %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %lui = riscv.lui 1 : -> x$ |
  // CHECK-NEXT: %auipc = riscv.auipc 1 : -> x$ |
  // CHECK-NEXT: %mv = riscv.mv %0 : x$ -> x$ |

  // Integer Register-Register Operations
  %add = "riscv.add"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %slt = "riscv.slt"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %sltu = "riscv.sltu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %and = "riscv.and"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %or = "riscv.or"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %xor = "riscv.xor"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %sll = "riscv.sll"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %srl = "riscv.srl"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %sub = "riscv.sub"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %sra = "riscv.sra"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  "riscv.nop"() : () -> ()
  // CHECK-NEXT: %add = riscv.add %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %slt = riscv.slt %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %sltu = riscv.sltu %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %and = riscv.and %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %or = riscv.or %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %xor = riscv.xor %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %sll = riscv.sll %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %srl = riscv.srl %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %sub = riscv.sub %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: %sra = riscv.sra %0, %1 : x$, x$ -> x$
  // CHECK-NEXT: riscv.nop : -> |

  // RV32I/RV64I: 2.5 Control Transfer Instructions
  // terminators continue at the end of module

  // Unconditional Branch Instructions
  "riscv.jal"() {"immediate" = 1 : i32} : () -> ()
  "riscv.jal"() {"immediate" = 1 : i32, "rd" = !riscv.reg<x$>} : () -> ()
  "riscv.jal"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: riscv.jal 1 : -> |
  // CHECK-NEXT: riscv.jal 1, x$ : -> |
  // CHECK-NEXT: riscv.jal "label" : -> |

  "riscv.j"() {"immediate" = 1 : i32} : () -> ()
  "riscv.j"() {"immediate" = #riscv.label<"label">} : () -> ()
  // CHECK-NEXT: riscv.j 1 : -> |
  // CHECK-NEXT: riscv.j "label" : -> |

  "riscv.jalr"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> ()
  "riscv.jalr"(%0) {"immediate" = 1 : i32, "rd" = !riscv.reg<x$>} : (!riscv.reg<x$>) -> ()
  "riscv.jalr"(%0) {"immediate" = #riscv.label<"label">} : (!riscv.reg<x$>) -> ()
  // CHECK-NEXT: riscv.jalr %0, 1 : x$ -> |
  // CHECK-NEXT: riscv.jalr %0, 1, x$ : x$ -> |
  // CHECK-NEXT: riscv.jalr %0, "label" : x$ -> |

  // Conditional Branch Instructions
  "riscv.beq"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.bne"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.blt"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.bge"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.bltu"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.bgeu"(%0, %1) {"offset" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: riscv.beq %0, %1, 1 : x$, x$ -> |
  // CHECK-NEXT: riscv.bne %0, %1, 1 : x$, x$ -> |
  // CHECK-NEXT: riscv.blt %0, %1, 1 : x$, x$ -> |
  // CHECK-NEXT: riscv.bge %0, %1, 1 : x$, x$ -> |
  // CHECK-NEXT: riscv.bltu %0, %1, 1 : x$, x$ -> |
  // CHECK-NEXT: riscv.bgeu %0, %1, 1 : x$, x$ -> |

  // RV32I/RV64I: 2.6 Load and Store Instructions

  %lb = "riscv.lb"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lbu = "riscv.lbu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lh = "riscv.lh"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lhu = "riscv.lhu"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %lw = "riscv.lw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  "riscv.sb"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.sh"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "riscv.sw"(%0, %1) {"immediate" = 1 : i32}: (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: %lb = riscv.lb %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %lbu = riscv.lbu %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %lh = riscv.lh %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %lhu = riscv.lhu %0, 1 : x$ -> x$ |
  // CHECK-NEXT: %lw = riscv.lw %0, 1 : x$ -> x$ |
  // CHECK-NEXT: riscv.sb %0, %1, 1 : x$, x$ -> |
  // CHECK-NEXT: riscv.sh %0, %1, 1 : x$, x$ -> |
  // CHECK-NEXT: riscv.sw %0, %1, 1 : x$, x$ -> |

  // RV32I/RV64I: 2.8 Control and Status Register Instructions

  %csrrw_rw = "riscv.csrrw"(%0) {"csr" = 1024 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrw_w = "riscv.csrrw"(%0) {"csr" = 1024 : i32, "writeonly"}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrs_rw = "riscv.csrrs"(%0) {"csr" = 1024 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrs_r = "riscv.csrrs"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrc_rw = "riscv.csrrc"(%0) {"csr" = 1024 : i32}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrc_r = "riscv.csrrc"(%0) {"csr" = 1024 : i32, "readonly"}: (!riscv.reg<x$>) -> !riscv.reg<x$>
  %csrrsi_rw = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<x$>
  %csrrsi_r = "riscv.csrrsi"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<x$>
  %csrrci_rw = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 8 : i32}: () -> !riscv.reg<x$>
  %csrrci_r = "riscv.csrrci"() {"csr" = 1024 : i32, "immediate" = 0 : i32}: () -> !riscv.reg<x$>
  %csrrwi_rw = "riscv.csrrwi"() {"csr" = 1024 : i32, "immediate" = 1 : i32}: () -> !riscv.reg<x$>
  %csrrwi_w = "riscv.csrrwi"() {"csr" = 1024 : i32, "writeonly", "immediate" = 1 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %csrrw_rw = riscv.csrrw %0, 1024 : x$ -> x$ |
  // CHECK-NEXT: %csrrw_w = riscv.csrrw %0, 1024, w : x$ -> x$ |
  // CHECK-NEXT: %csrrs_rw = riscv.csrrs %0, 1024 : x$ -> x$ |
  // CHECK-NEXT: %csrrs_r = riscv.csrrs %0, 1024, r : x$ -> x$ |
  // CHECK-NEXT: %csrrc_rw = riscv.csrrc %0, 1024 : x$ -> x$ |
  // CHECK-NEXT: %csrrc_r = riscv.csrrc %0, 1024, r : x$ -> x$ |
  // CHECK-NEXT: %csrrsi_rw = riscv.csrrsi 1024, 8 : -> x$ |
  // CHECK-NEXT: %csrrsi_r = riscv.csrrsi 1024, 0 : -> x$ |
  // CHECK-NEXT: %csrrci_rw = riscv.csrrci 1024, 8 : -> x$ |
  // CHECK-NEXT: %csrrci_r = riscv.csrrci 1024, 0 : -> x$ |
  // CHECK-NEXT: %csrrwi_rw = riscv.csrrwi 1024, 1 : -> x$ |
  // CHECK-NEXT: %csrrwi_w = riscv.csrrwi 1024, w, 1 : -> x$ |

  // Machine Mode Privileged Instructions
  "riscv.wfi"() : () -> ()
  // CHECK-NEXT: riscv.wfi : -> |


  // RV32M/RV64M: 7 “M” Standard Extension for Integer Multiplication and Division

  // Multiplication Operations
  %mul = "riscv.mul"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %mulh = "riscv.mulh"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %mulhsu = "riscv.mulhsu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %mulhu = "riscv.mulhu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %mul = riscv.mul %0, %1 : x$, x$ -> x$ |
  // CHECK-NEXT: %mulh = riscv.mulh %0, %1 : x$, x$ -> x$ |
  // CHECK-NEXT: %mulhsu = riscv.mulhsu %0, %1 : x$, x$ -> x$ |
  // CHECK-NEXT: %mulhu = riscv.mulhu %0, %1 : x$, x$ -> x$ |

  // Division Operations
  %div = "riscv.div"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %divu = "riscv.divu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %rem = "riscv.rem"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  %remu = "riscv.remu"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %div = riscv.div %0, %1 : x$, x$ -> x$ |
  // CHECK-NEXT: %divu = riscv.divu %0, %1 : x$, x$ -> x$ |
  // CHECK-NEXT: %rem = riscv.rem %0, %1 : x$, x$ -> x$ |
  // CHECK-NEXT: %remu = riscv.remu %0, %1 : x$, x$ -> x$ |

  // Assembler pseudo-instructions

  %li = "riscv.li"() {"immediate" = 1 : i32}: () -> !riscv.reg<x$>
  // CHECK-NEXT: %li = riscv.li 1 : -> x$ |

  // Environment Call and Breakpoints
  "riscv.ecall"() : () -> ()
  "riscv.ebreak"() : () -> ()
  "riscv.directive"() {"directive" = ".align", "value" = "2"} : () -> ()
  "riscv.directive"() ({
    %nested_li = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
  }) {"directive" = ".text"} : () -> ()
  // CHECK-NEXT: riscv.ecall : -> |
  // CHECK-NEXT: riscv.ebreak : -> |
  // CHECK-NEXT: riscv.directive ".align", "2" : -> |
  // CHECK-NEXT: riscv.directive ".text" ({
  // CHECK-NEXT:   %nested_li = riscv.li 1 : -> x$ |
  // CHECK-NEXT: }) : -> |

  // Custom instruction
  %custom0, %custom1 = "riscv.custom_assembly_instruction"(%0, %1) {"instruction_name" = "hello"} : (!riscv.reg<x$>, !riscv.reg<x$>) -> (!riscv.reg<x$>, !riscv.reg<x$>)
  // CHECK-NEXT: %custom0, %custom1 = riscv.custom_assembly_instruction %0, %1, "hello" : x$, x$ -> x$, x$ |


  // RISC-V extensions
  "riscv.scfgw"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK-NEXT: riscv.scfgw %0, %1 : x$, x$ -> |

  // RV32I/RV64I: 2.5 Control Transfer Instructions (cont'd)
  // terminators

  // RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0
  %f0 = "riscv.get_float_register"() : () -> !riscv.freg<f$>
  %f1 = "riscv.get_float_register"() : () -> !riscv.freg<f$>
  %f2 = "riscv.get_float_register"() : () -> !riscv.freg<f$>
  // CHECK-NEXT: %f0 = riscv.get_float_register : -> f$ |
  // CHECK-NEXT: %f1 = riscv.get_float_register : -> f$ |
  // CHECK-NEXT: %f2 = riscv.get_float_register : -> f$ |

  %fmadd_s = "riscv.fmadd.s"(%f0, %f1, %f2) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fmsub_s = "riscv.fmsub.s"(%f0, %f1, %f2) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fnmsub_s = "riscv.fnmsub.s"(%f0, %f1, %f2) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fnmadd_s = "riscv.fnmadd.s"(%f0, %f1, %f2) : (!riscv.freg<f$>, !riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %fmadd_s = riscv.fmadd.s %f0, %f1, %f2 : f$, f$, f$ -> f$ |
  // CHECK-NEXT: %fmsub_s = riscv.fmsub.s %f0, %f1, %f2 : f$, f$, f$ -> f$ |
  // CHECK-NEXT: %fnmsub_s = riscv.fnmsub.s %f0, %f1, %f2 : f$, f$, f$ -> f$ |
  // CHECK-NEXT: %fnmadd_s = riscv.fnmadd.s %f0, %f1, %f2 : f$, f$, f$ -> f$ |

  %fadd_s = "riscv.fadd.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fsub_s = "riscv.fsub.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fmul_s = "riscv.fmul.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fdiv_s = "riscv.fdiv.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fsqrt_s = "riscv.fsqrt.s"(%f0) : (!riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %fadd_s = riscv.fadd.s %f0, %f1 : f$, f$ -> f$ |
  // CHECK-NEXT: %fsub_s = riscv.fsub.s %f0, %f1 : f$, f$ -> f$ |
  // CHECK-NEXT: %fmul_s = riscv.fmul.s %f0, %f1 : f$, f$ -> f$ |
  // CHECK-NEXT: %fdiv_s = riscv.fdiv.s %f0, %f1 : f$, f$ -> f$ |
  // CHECK-NEXT: %fsqrt_s = riscv.fsqrt.s %f0 : f$ -> f$ |

  %fsgnj_s = "riscv.fsgnj.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fsgnjn_s = "riscv.fsgnjn.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fsgnjx_s = "riscv.fsgnjx.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %fsgnj_s = riscv.fsgnj.s %f0, %f1 : f$, f$ -> f$ |
  // CHECK-NEXT: %fsgnjn_s = riscv.fsgnjn.s %f0, %f1 : f$, f$ -> f$ |
  // CHECK-NEXT: %fsgnjx_s = riscv.fsgnjx.s %f0, %f1 : f$, f$ -> f$ |

  %fmin_s = "riscv.fmin.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  %fmax_s = "riscv.fmax.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %fmin_s = riscv.fmin.s %f0, %f1 : f$, f$ -> f$ |
  // CHECK-NEXT: %fmax_s = riscv.fmax.s %f0, %f1 : f$, f$ -> f$ |

  %fcvt_w_s = "riscv.fcvt.w.s"(%f0) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  %fcvt_wu_s = "riscv.fcvt.wu.s"(%f0) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  %fmv_x_w = "riscv.fmv.x.w"(%f0) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  // CHECK-NEXT: %fcvt_w_s = riscv.fcvt.w.s %f0 : f$ -> x$ |
  // CHECK-NEXT: %fcvt_wu_s = riscv.fcvt.wu.s %f0 : f$ -> x$ |
  // CHECK-NEXT: %fmv_x_w = riscv.fmv.x.w %f0 : f$ -> x$ |

  %feq_s = "riscv.feq.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  %flt_s = "riscv.flt.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  %fle_s = "riscv.fle.s"(%f0, %f1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
  %fclass_s = "riscv.fclass.s"(%f0) : (!riscv.freg<f$>) -> !riscv.reg<x$>
  %fcvt_s_w = "riscv.fcvt.s.w"(%0) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  %fcvt_s_wu = "riscv.fcvt.s.wu"(%0) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  %fmv_w_x = "riscv.fmv.w.x"(%0) : (!riscv.reg<x$>) -> !riscv.freg<f$>
  // CHECK-NEXT: %feq_s = riscv.feq.s %f0, %f1 : f$, f$ -> x$ |
  // CHECK-NEXT: %flt_s = riscv.flt.s %f0, %f1 : f$, f$ -> x$ |
  // CHECK-NEXT: %fle_s = riscv.fle.s %f0, %f1 : f$, f$ -> x$ |
  // CHECK-NEXT: %fclass_s = riscv.fclass.s %f0 : f$ -> x$ |
  // CHECK-NEXT: %fcvt_s_w = riscv.fcvt.s.w %0 : x$ -> f$ |
  // CHECK-NEXT: %fcvt_s_wu = riscv.fcvt.s.wu %0 : x$ -> f$ |
  // CHECK-NEXT: %fmv_w_x = riscv.fmv.w.x %0 : x$ -> f$ |

  %flw = "riscv.flw"(%0) {"immediate" = 1 : i32}: (!riscv.reg<x$>) -> !riscv.freg<f$>
  "riscv.fsw"(%0, %f0) {"immediate" = 1 : i32} : (!riscv.reg<x$>, !riscv.freg<f$>) -> ()
  // CHECK-NEXT: %flw = riscv.flw %0, 1 : x$ -> f$ |
  // CHECK-NEXT: riscv.fsw %0, %f0, 1 : x$, f$ -> |

  "riscv.jal"() {"immediate" = 1 : i32, "test" = "hello", "comment" = "world"} : () -> ()
  // CHECK-NEXT: riscv.jal 1 {"test" = "hello", "comment" = "world"} : -> |

  // Unconditional Branch Instructions
  "riscv.ret"() : () -> ()
  // CHECK-NEXT: riscv.ret : -> |
}) : () -> ()
