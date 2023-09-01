// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=BlockNaive exclude_preallocated=true}" %s | filecheck %s --check-prefix=BNAIVE
// RUN: xdsl-opt -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive exclude_preallocated=true}" %s | filecheck %s --check-prefix=LIVE-BNAIVE

riscv_func.func @main() {
  %0 = riscv.li 6 : () -> !riscv.reg<>
  %1 = riscv.li 5 : () -> !riscv.reg<t6>
  %3 = riscv.fcvt.s.w %0 : (!riscv.reg<>) -> !riscv.freg<>
  %4 = riscv.fcvt.s.w %1 : (!riscv.reg<t6>) -> !riscv.freg<>
  %5 = riscv.fadd.s %3, %4 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %2 = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<t6>) -> !riscv.reg<>
  riscv_func.return
}

// BNAIVE:       builtin.module {
// BNAIVE-NEXT:    riscv_func.func @main() {
// BNAIVE-NEXT:      %0 = riscv.li 6 : () -> !riscv.reg<t5>
// BNAIVE-NEXT:      %1 = riscv.li 5 : () -> !riscv.reg<t6>
// BNAIVE-NEXT:      %2 = riscv.fcvt.s.w %0 : (!riscv.reg<t5>) -> !riscv.freg<ft11>
// BNAIVE-NEXT:      %3 = riscv.fcvt.s.w %1 : (!riscv.reg<t6>) -> !riscv.freg<ft10>
// BNAIVE-NEXT:      %4 = riscv.fadd.s %2, %3 : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft9>
// BNAIVE-NEXT:      %5 = riscv.add %0, %1 : (!riscv.reg<t5>, !riscv.reg<t6>) -> !riscv.reg<t4>
// BNAIVE-NEXT:      riscv_func.return
// BNAIVE-NEXT:    }
// BNAIVE-NEXT:  }

// LIVE-BNAIVE:       builtin.module {
// LIVE-BNAIVE-NEXT:    riscv_func.func @main() {
// LIVE-BNAIVE-NEXT:      %0 = riscv.li 6 : () -> !riscv.reg<t5>
// LIVE-BNAIVE-NEXT:      %1 = riscv.li 5 : () -> !riscv.reg<t6>
// LIVE-BNAIVE-NEXT:      %2 = riscv.fcvt.s.w %0 : (!riscv.reg<t5>) -> !riscv.freg<ft11>
// LIVE-BNAIVE-NEXT:      %3 = riscv.fcvt.s.w %1 : (!riscv.reg<t6>) -> !riscv.freg<ft10>
// LIVE-BNAIVE-NEXT:      %4 = riscv.fadd.s %2, %3 : (!riscv.freg<ft11>, !riscv.freg<ft10>) -> !riscv.freg<ft11>
// LIVE-BNAIVE-NEXT:      %5 = riscv.add %0, %1 : (!riscv.reg<t5>, !riscv.reg<t6>) -> !riscv.reg<t5>
// LIVE-BNAIVE-NEXT:      riscv_func.return
// LIVE-BNAIVE-NEXT:    }
// LIVE-BNAIVE-NEXT:  }
