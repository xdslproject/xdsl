// RUN: xdsl-opt --split-input-file -t riscemu %s | filecheck %s

builtin.module {
  %0 = "riscv.li"() {"immediate" = 6 : si32} : () -> !riscv.reg<j0>
  %1 = "riscv.li"() {"immediate" = 7 : si32} : () -> !riscv.reg<j1>
  %2 = "riscv.mul"(%0, %1) : (!riscv.reg<j0>, !riscv.reg<j1>) -> !riscv.reg<j2>
  "riscv.custom_assembly_instruction"(%2) {"instruction_name" = "print"} : (!riscv.reg<j2>) -> ()
  %3 = "riscv.li"() {"immediate" = 93 : si32} : () -> !riscv.reg<a7>
  "riscv.ecall"() : () -> ()
  "riscv.ret"() : () -> ()
}

// CHECK: 42

// -----

builtin.module {
  "riscv.label"() ({
    %0 = "riscv.li"() {"immediate" = 3 : si32} : () -> !riscv.reg<a0>
    %1 = "riscv.li"() {"immediate" = 2 : si32} : () -> !riscv.reg<a1>
    %2 = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<a2>
    "riscv.jal"() {"immediate" = #riscv.label<"muladd">} : () -> ()
    %3 = "riscv.get_register"() : () -> !riscv.reg<a0>
    "riscv.custom_assembly_instruction"(%3) {"instruction_name" = "print"} : (!riscv.reg<a0>) -> ()
    %4 = "riscv.li"() {"immediate" = 93 : si32} : () -> !riscv.reg<a7>
    "riscv.ecall"() : () -> ()
    "riscv.ret"() : () -> ()
  }) {"label" = #riscv.label<"main">} : () -> ()
  "riscv.label"() ({
    "riscv.comment"() {"comment" = "no extra registers needed, so no need to deal with stack"} : () -> ()
    %5 = "riscv.get_register"() : () -> !riscv.reg<a0>
    %6 = "riscv.get_register"() : () -> !riscv.reg<a1>
    %7 = "riscv.mul"(%5, %6) : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
    "riscv.ret"() : () -> ()
  }) {"label" = #riscv.label<"multiply">} : () -> ()
  "riscv.label"() ({
    "riscv.comment"() {"comment" = "no extra registers needed, so no need to deal with stack"} : () -> ()
    %8 = "riscv.get_register"() : () -> !riscv.reg<a0>
    %9 = "riscv.get_register"() : () -> !riscv.reg<a1>
    %10 = "riscv.add"(%8, %9) : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
    "riscv.ret"() : () -> ()
  }) {"label" = #riscv.label<"add">} : () -> ()
  "riscv.label"() ({
    "riscv.comment"() {"comment" = "a0 <- a0 * a1 + a2"} : () -> ()
    "riscv.comment"() {"comment" = "prologue"} : () -> ()
    %11 = "riscv.get_register"() : () -> !riscv.reg<a2>
    %12 = "riscv.get_register"() : () -> !riscv.reg<sp>
    %13 = "riscv.get_register"() : () -> !riscv.reg<s0>
    %14 = "riscv.get_register"() : () -> !riscv.reg<ra>
    "riscv.comment"() {"comment" = "decrement stack pointer by number of register values we need to store for later"} : () -> ()
    %15 = "riscv.addi"(%12) {"immediate" = -8 : si12} : (!riscv.reg<sp>) -> !riscv.reg<sp>
    "riscv.comment"() {"comment" = "save the s registers we'll use on the stack"} : () -> ()
    "riscv.sw"(%12, %13) {"immediate" = 0 : si12} : (!riscv.reg<sp>, !riscv.reg<s0>) -> ()
    "riscv.comment"() {"comment" = "save the return address we'll use on the stack"} : () -> ()
    "riscv.sw"(%12, %14) {"immediate" = 4 : si12} : (!riscv.reg<sp>, !riscv.reg<ra>) -> ()
    %16 = "riscv.mv"(%11) : (!riscv.reg<a2>) -> !riscv.reg<s0>
    "riscv.jal"() {"immediate" = #riscv.label<"multiply">} : () -> ()
    %17 = "riscv.mv"(%16) : (!riscv.reg<s0>) -> !riscv.reg<a1>
    "riscv.jal"() {"immediate" = #riscv.label<"add">} : () -> ()
    "riscv.comment"() {"comment" = "epilogue"} : () -> ()
    "riscv.comment"() {"comment" = "store the old values back into the s registers"} : () -> ()
    %18 = "riscv.lw"(%12) {"immediate" = 0 : si12} : (!riscv.reg<sp>) -> !riscv.reg<s0>
    "riscv.comment"() {"comment" = "store the return address back into the ra register"} : () -> ()
    %19 = "riscv.lw"(%12) {"immediate" = 4 : si12} : (!riscv.reg<sp>) -> !riscv.reg<ra>
    "riscv.comment"() {"comment" = "set the sp back to what it was at the start of the function call"} : () -> ()
    %20 = "riscv.addi"(%12) {"immediate" = 8 : si12} : (!riscv.reg<sp>) -> !riscv.reg<sp>
    "riscv.comment"() {"comment" = "jump back to caller"} : () -> ()
    "riscv.ret"() : () -> ()
  }) {"label" = #riscv.label<"muladd">} : () -> ()
}

// CHECK: 7
