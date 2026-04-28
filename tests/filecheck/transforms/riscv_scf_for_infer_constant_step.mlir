// RUN: xdsl-opt -p riscv-scf-for-infer-constant-step %s | filecheck %s

%lb, %ub, %step_dynamic = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
%step_static = rv32.li 4 : !riscv.reg
riscv_scf.for %i_static : !riscv.reg = %lb to %ub step %step_static {
  riscv_scf.yield
}
// CHECK:    riscv_scf.for %i_static : !riscv.reg = %lb to %ub step 4 : si12 {


riscv_scf.for %i_dynamic : !riscv.reg = %lb to %ub step %step_dynamic {
  riscv_scf.yield
}

// CHECK:    riscv_scf.for %i_dynamic : !riscv.reg = %lb to %ub step %step_dynamic {
