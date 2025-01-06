// RUN: xdsl-opt -p lower-riscv-scf-to-labels -t riscemu %s
// RUN: xdsl-opt -p convert-riscv-scf-to-riscv-cf -t riscemu %s

builtin.module {
  riscv_func.func @main() {
    %0 = riscv.li 0 : !riscv.reg<a0>
    %1 = riscv.li 10 : !riscv.reg<a1>
    %2 = riscv.li 1 : !riscv.reg<a2>
    %3 = riscv.li 0 : !riscv.reg<a3>
    %4 = riscv_scf.for %5 : !riscv.reg<a0> = %0 to %1 step %2 iter_args(%6 = %3) -> (!riscv.reg<a3>) {
      %7 = riscv.add %5, %6 : (!riscv.reg<a0>, !riscv.reg<a3>) -> !riscv.reg<a3>
      riscv_scf.yield %7 : !riscv.reg<a3>
    }
    %8 = riscv.mv %4 : (!riscv.reg<a3>) -> !riscv.reg<a0>
    riscv.custom_assembly_instruction %4 {"instruction_name" = "print"} : (!riscv.reg<a3>) -> ()
    riscv.li 93 : !riscv.reg<a7>
    riscv.ecall
    riscv_func.return
  }
}

// CHECK: 45
