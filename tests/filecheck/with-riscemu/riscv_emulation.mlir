// RUN: xdsl-opt --split-input-file -t riscemu %s | filecheck %s

builtin.module {
  riscv.directive ".globl" "main"
  riscv_func.func @main() {
    %0 = riscv.li 6 : () -> !riscv.reg<j0>
    %1 = riscv.li 7 : () -> !riscv.reg<j1>
    %2 = riscv.mul %0, %1 : (!riscv.reg<j0>, !riscv.reg<j1>) -> !riscv.reg<j2>
    riscv.custom_assembly_instruction %2 {"instruction_name" = "print"} : (!riscv.reg<j2>) -> ()
    %3 = riscv.li 93 : () -> !riscv.reg<a7>
    riscv.ecall
    riscv_func.return
  }
}

// CHECK: 42

// -----

builtin.module {
  riscv.directive ".globl" "main"
  riscv_func.func @main() {
    %0 = riscv.li 3 : () -> !riscv.reg<a0>
    %1 = riscv.li 2 : () -> !riscv.reg<a1>
    %2 = riscv.li 1 : () -> !riscv.reg<a2>
    %xyz = riscv_func.call @muladd(%x, %y, %z) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>) -> !riscv.reg<a0>
    riscv.custom_assembly_instruction %xyz {"instruction_name" = "print"} : (!riscv.reg<a0>) -> ()
    %4 = riscv.li 93 : () -> !riscv.reg<a7>
    riscv.ecall
    riscv_func.return
  }
  riscv_func.func @multiply(%x : !riscv.reg<a0>, %y : !riscv.reg<a1>) {
    "riscv.comment"() {"comment" = "no extra registers needed, so no need to deal with stack"} : () -> ()
    %product = riscv.mul %x, %y : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
    riscv_func.return %product : !riscv.reg<a0>
  }
  riscv_func.func @add(%x : !riscv.reg<a0>, %y : !riscv.reg<a1>) {
    "riscv.comment"() {"comment" = "no extra registers needed, so no need to deal with stack"} : () -> ()
    %sum = "riscv.add"(%x, %y) : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
    riscv_func.return %sum : !riscv.reg<a0>
  }
  riscv_func.func @muladd(%x : !riscv.reg<a0>, %y : !riscv.reg<a1>, %z : !riscv.reg<a2>) {
    "riscv.comment"() {"comment" = "a0 <- a0 * a1 + a2"} : () -> ()
    "riscv.comment"() {"comment" = "prologue"} : () -> ()
    %12 = riscv.get_register : () -> !riscv.reg<sp>
    %13 = riscv.get_register : () -> !riscv.reg<s0>
    %14 = riscv.get_register : () -> !riscv.reg<ra>
    "riscv.comment"() {"comment" = "decrement stack pointer by number of register values we need to store for later"} : () -> ()
    %15 = riscv.addi %12, -8 : (!riscv.reg<sp>) -> !riscv.reg<sp>
    "riscv.comment"() {"comment" = "save the s registers we'll use on the stack"} : () -> ()
    riscv.sw %12, %13, 0: (!riscv.reg<sp>, !riscv.reg<s0>) -> ()
    "riscv.comment"() {"comment" = "save the return address we'll use on the stack"} : () -> ()
    riscv.sw %12, %14, 4: (!riscv.reg<sp>, !riscv.reg<ra>) -> ()
    %16 = riscv.mv %z : (!riscv.reg<a2>) -> !riscv.reg<s0>
    %xy = riscv_func.call @multiply(%x, %y) : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
    %17 = riscv.mv %16 : (!riscv.reg<s0>) -> !riscv.reg<a1>
    %xyz = riscv_func.call @add(%xy, %17) : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
    "riscv.comment"() {"comment" = "epilogue"} : () -> ()
    "riscv.comment"() {"comment" = "store the old values back into the s registers"} : () -> ()
    %18 = riscv.lw %12, 0: (!riscv.reg<sp>) -> !riscv.reg<s0>
    "riscv.comment"() {"comment" = "store the return address back into the ra register"} : () -> ()
    %19 = riscv.lw %12, 4: (!riscv.reg<sp>) -> !riscv.reg<ra>
    "riscv.comment"() {"comment" = "set the sp back to what it was at the start of the function call"} : () -> ()
    %20 = riscv.addi %12, 8: (!riscv.reg<sp>) -> !riscv.reg<sp>
    "riscv.comment"() {"comment" = "jump back to caller"} : () -> ()
    riscv_func.return %xyz : !riscv.reg<a0>
  }
}

// CHECK: 7
