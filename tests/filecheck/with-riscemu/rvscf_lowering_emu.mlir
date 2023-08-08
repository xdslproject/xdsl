// RUN: xdsl-opt -p lower-riscv-scf-to-labels -t riscemu %s

builtin.module {
  riscv.label {"label" = #riscv.label<"main">} ({
    %0 = riscv.li 0 : () -> !riscv.reg<a0>
    %1 = riscv.li 10 : () -> !riscv.reg<a1>
    %2 = riscv.li 1 : () -> !riscv.reg<a2>
    %3 = riscv.li 0 : () -> !riscv.reg<a3>
    %4 = "riscv_scf.for"(%0, %1, %2, %3) ({
    ^0(%5 : !riscv.reg<a0>, %6 : !riscv.reg<a3>):
      %7 = riscv.add %5, %6 : (!riscv.reg<a0>, !riscv.reg<a3>) -> !riscv.reg<a3>
      "riscv_scf.yield"(%7) : (!riscv.reg<a3>) -> ()
    }) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>) -> !riscv.reg<a3>
    %8 = riscv.mv %4 : (!riscv.reg<a3>) -> !riscv.reg<a0>
    "riscv.custom_assembly_instruction"(%4) {"instruction_name" = "print"} : (!riscv.reg<a3>) -> ()
    riscv.li 93 : () -> !riscv.reg<a7>
    riscv.ecall : () -> ()
    riscv.ret : () -> ()
  }) : () -> ()
}

// CHECK: 45