// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=GlobalJRegs exclude_preallocated=true}" %s --print-op-generic | filecheck %s --check-prefix=GJREGS
// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=BlockNaive exclude_preallocated=true}" %s --print-op-generic | filecheck %s --check-prefix=BNAIVE
// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive exclude_preallocated=true}" %s --print-op-generic | filecheck %s --check-prefix=LIVE-BNAIVE

%0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<>
%1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<t6>
%3 = "riscv.fcvt.s.w"(%0) : (!riscv.reg<>) -> !riscv.freg<>
%4 = "riscv.fcvt.s.w"(%1) : (!riscv.reg<t6>) -> !riscv.freg<>
%5 = "riscv.fadd.s"(%3, %4) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
%2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<t6>) -> !riscv.reg<>

// GJREGS: %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<j0>
// GJREGS: %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<t6>
// GJREGS: %2 = "riscv.fcvt.s.w"(%0) : (!riscv.reg<j0>) -> !riscv.freg<j1>
// GJREGS: %3 = "riscv.fcvt.s.w"(%1) : (!riscv.reg<t6>) -> !riscv.freg<j2>
// GJREGS: %4 = "riscv.fadd.s"(%2, %3) : (!riscv.freg<j1>, !riscv.freg<j2>) -> !riscv.freg<j3>
// GJREGS: %5 = "riscv.add"(%0, %1) : (!riscv.reg<j0>, !riscv.reg<t6>) -> !riscv.reg<j4>

// BNAIVE: %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<t5>
// BNAIVE: %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<t6>
// BNAIVE: %2 = "riscv.fcvt.s.w"(%0) : (!riscv.reg<t5>) -> !riscv.freg<ft11>
// BNAIVE: %3 = "riscv.fcvt.s.w"(%1) : (!riscv.reg<t6>) -> !riscv.freg<ft10>
// BNAIVE: %4 = "riscv.fadd.s"(%2, %3) : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft9>
// BNAIVE: %5 = "riscv.add"(%0, %1) : (!riscv.reg<t5>, !riscv.reg<t6>) -> !riscv.reg<t4>

// LIVE-BNAIVE: %0 = "riscv.li"() {"immediate" = 6 : i32} : () -> !riscv.reg<t5>
// LIVE-BNAIVE: %1 = "riscv.li"() {"immediate" = 5 : i32} : () -> !riscv.reg<t6>
// LIVE-BNAIVE: %2 = "riscv.fcvt.s.w"(%0) : (!riscv.reg<t5>) -> !riscv.freg<ft11>
// LIVE-BNAIVE: %3 = "riscv.fcvt.s.w"(%1) : (!riscv.reg<t6>) -> !riscv.freg<ft10>
// LIVE-BNAIVE: %4 = "riscv.fadd.s"(%2, %3) : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft9>
// LIVE-BNAIVE: %5 = "riscv.add"(%0, %1) : (!riscv.reg<t5>, !riscv.reg<t6>) -> !riscv.reg<t4>
