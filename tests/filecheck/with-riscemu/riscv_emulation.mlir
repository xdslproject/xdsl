// RUN: xdsl-opt --split-input-file -t riscemu %s | filecheck %s

builtin.module {
  %0 = "riscv.li"() {"immediate" = 6 : si32} : () -> !riscv.ireg<j0>
  %1 = "riscv.li"() {"immediate" = 7 : si32} : () -> !riscv.ireg<j1>
  %2 = "riscv.mul"(%0, %1) : (!riscv.ireg<j0>, !riscv.ireg<j1>) -> !riscv.ireg<j2>
  "riscv.custom_assembly_instruction"(%2) {"instruction_name" = "print"} : (!riscv.ireg<j2>) -> ()
  %3 = "riscv.li"() {"immediate" = 93 : si32} : () -> !riscv.ireg<a7>
  "riscv.ecall"() : () -> ()
  "riscv.ret"() : () -> ()
}

// CHECK: 42

// -----

builtin.module {
  %0 = "riscv.li"() {"immediate" = 1084227584 : si32} : () -> !riscv.ireg<j0>
  %1 = "riscv.li"() {"immediate" = 1082130432 : si32} : () -> !riscv.ireg<j1>
  %2 = "riscv.fcvt.s.wu"(%0) : (!riscv.ireg<j0>) -> !riscv.freg<ft11>
  %3 = "riscv.fcvt.s.wu"(%1) : (!riscv.ireg<j1>) -> !riscv.freg<ft10>
  %4 = "riscv.fmul.s"(%2, %3) : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft9>
  "riscv.custom_assembly_instruction"(%4) {"instruction_name" = "print.float"} : (!riscv.freg<ft9>) -> ()
  %5 = "riscv.li"() {"immediate" = 93 : si32} : () -> !riscv.ireg<a7>
  "riscv.ecall"() : () -> ()
  "riscv.ret"() : () -> ()
}

// CHECK: 20.0

// -----

builtin.module {
  "riscv.label"() ({
    %0 = "riscv.li"() {"immediate" = 3 : si32} : () -> !riscv.ireg<a0>
    %1 = "riscv.li"() {"immediate" = 2 : si32} : () -> !riscv.ireg<a1>
    %2 = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.ireg<a2>
    "riscv.jal"() {"immediate" = #riscv.label<"muladd">} : () -> ()
    %3 = "riscv.get_integer_register"() : () -> !riscv.ireg<a0>
    "riscv.custom_assembly_instruction"(%3) {"instruction_name" = "print"} : (!riscv.ireg<a0>) -> ()
    %4 = "riscv.li"() {"immediate" = 93 : si32} : () -> !riscv.ireg<a7>
    "riscv.ecall"() : () -> ()
  }) {"label" = #riscv.label<"main">} : () -> ()
  "riscv.label"() ({
    "riscv.comment"() {"comment" = "no extra registers needed, so no need to deal with stack"} : () -> ()
    %5 = "riscv.get_integer_register"() : () -> !riscv.ireg<a0>
    %6 = "riscv.get_integer_register"() : () -> !riscv.ireg<a1>
    %7 = "riscv.mul"(%5, %6) : (!riscv.ireg<a0>, !riscv.ireg<a1>) -> !riscv.ireg<a0>
    "riscv.ret"() : () -> ()
  }) {"label" = #riscv.label<"multiply">} : () -> ()
  "riscv.label"() ({
    "riscv.comment"() {"comment" = "no extra registers needed, so no need to deal with stack"} : () -> ()
    %8 = "riscv.get_integer_register"() : () -> !riscv.ireg<a0>
    %9 = "riscv.get_integer_register"() : () -> !riscv.ireg<a1>
    %10 = "riscv.add"(%8, %9) : (!riscv.ireg<a0>, !riscv.ireg<a1>) -> !riscv.ireg<a0>
    "riscv.ret"() : () -> ()
  }) {"label" = #riscv.label<"add">} : () -> ()
  "riscv.label"() ({
    "riscv.comment"() {"comment" = "a0 <- a0 * a1 + a2"} : () -> ()
    "riscv.comment"() {"comment" = "prologue"} : () -> ()
    %11 = "riscv.get_integer_register"() : () -> !riscv.ireg<a2>
    %12 = "riscv.get_integer_register"() : () -> !riscv.ireg<sp>
    %13 = "riscv.get_integer_register"() : () -> !riscv.ireg<s0>
    %14 = "riscv.get_integer_register"() : () -> !riscv.ireg<ra>
    "riscv.comment"() {"comment" = "decrement stack pointer by number of register values we need to store for later"} : () -> ()
    %15 = "riscv.addi"(%12) {"immediate" = -8 : si12} : (!riscv.ireg<sp>) -> !riscv.ireg<sp>
    "riscv.comment"() {"comment" = "save the s registers we'll use on the stack"} : () -> ()
    "riscv.sw"(%13, %12) {"immediate" = 0 : si12} : (!riscv.ireg<s0>, !riscv.ireg<sp>) -> ()
    "riscv.comment"() {"comment" = "save the return address we'll use on the stack"} : () -> ()
    "riscv.sw"(%14, %12) {"immediate" = 4 : si12} : (!riscv.ireg<ra>, !riscv.ireg<sp>) -> ()
    %16 = "riscv.mv"(%11) : (!riscv.ireg<a2>) -> !riscv.ireg<s0>
    "riscv.jal"() {"immediate" = #riscv.label<"multiply">} : () -> ()
    %17 = "riscv.mv"(%16) : (!riscv.ireg<s0>) -> !riscv.ireg<a1>
    "riscv.jal"() {"immediate" = #riscv.label<"add">} : () -> ()
    "riscv.comment"() {"comment" = "epilogue"} : () -> ()
    "riscv.comment"() {"comment" = "store the old values back into the s registers"} : () -> ()
    %18 = "riscv.lw"(%12) {"immediate" = 0 : si12} : (!riscv.ireg<sp>) -> !riscv.ireg<s0>
    "riscv.comment"() {"comment" = "store the return address back into the ra register"} : () -> ()
    %19 = "riscv.lw"(%12) {"immediate" = 4 : si12} : (!riscv.ireg<sp>) -> !riscv.ireg<ra>
    "riscv.comment"() {"comment" = "set the sp back to what it was at the start of the function call"} : () -> ()
    %20 = "riscv.addi"(%12) {"immediate" = 8 : si12} : (!riscv.ireg<sp>) -> !riscv.ireg<sp>
    "riscv.comment"() {"comment" = "jump back to caller"} : () -> ()
    "riscv.ret"() : () -> ()
  }) {"label" = #riscv.label<"muladd">} : () -> ()
}

// CHECK: 7
