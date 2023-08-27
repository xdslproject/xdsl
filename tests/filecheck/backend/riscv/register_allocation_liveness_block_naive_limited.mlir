// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive limit_registers=2}" %s | filecheck %s

%0 = riscv.li  6 : () -> !riscv.reg<>
%1 = riscv.li  5 : () -> !riscv.reg<s0>
%2 = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<s0>) -> !riscv.reg<>
%3 = riscv.li  29 : () -> !riscv.reg<>
%4 = riscv.li  28 : () -> !riscv.reg<>
%5 = riscv.add %3, %4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
%6 = riscv.li  26 : () -> !riscv.reg<>
%7 = riscv.li  25 : () -> !riscv.reg<>
%8 = riscv.li  24 : () -> !riscv.reg<>
%9 = riscv.li  23 : () -> !riscv.reg<>
%10 = riscv.li 2 : () -> !riscv.reg<>
%11 = riscv.li 1 : () -> !riscv.reg<>
%12 = riscv.fcvt.s.w %10 : (!riscv.reg<>) -> !riscv.freg<>
%13 = riscv.fcvt.s.w %11 : (!riscv.reg<>) -> !riscv.freg<>
%14 = riscv.fadd.s %12, %13 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

// CHECK: builtin.module {
// CHECK:   %0 = riscv.li 6 : () -> !riscv.reg<ra>
// CHECK:   %1 = riscv.li 5 : () -> !riscv.reg<s0>
// CHECK:   %2 = riscv.add %0, %1 : (!riscv.reg<ra>, !riscv.reg<s0>) -> !riscv.reg<t0>
// CHECK:   %3 = riscv.li 29 : () -> !riscv.reg<ra>
// CHECK:   %4 = riscv.li 28 : () -> !riscv.reg<t0>
// CHECK:   %5 = riscv.add %3, %4 : (!riscv.reg<ra>, !riscv.reg<t0>) -> !riscv.reg<j1>
// CHECK:   %6 = riscv.li 26 : () -> !riscv.reg<ra>
// CHECK:   %7 = riscv.li 25 : () -> !riscv.reg<ra>
// CHECK:   %8 = riscv.li 24 : () -> !riscv.reg<ra>
// CHECK:   %9 = riscv.li 23 : () -> !riscv.reg<ra>
// CHECK:   %10 = riscv.li 2 : () -> !riscv.reg<ra>
// CHECK:   %11 = riscv.li 1 : () -> !riscv.reg<t0>
// CHECK:   %12 = riscv.fcvt.s.w %10 : (!riscv.reg<ra>) -> !riscv.freg<ft1>
// CHECK:   %13 = riscv.fcvt.s.w %11 : (!riscv.reg<t0>) -> !riscv.freg<ft0>
// CHECK:   %14 = riscv.fadd.s %12, %13 : (!riscv.freg<ft1>, !riscv.freg<ft0>) -> !riscv.freg<j0>
// CHECK: }
