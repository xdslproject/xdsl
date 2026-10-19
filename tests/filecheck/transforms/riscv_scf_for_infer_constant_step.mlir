// RUN: xdsl-opt -p riscv-scf-for-infer-constant-step %s | filecheck %s

%lb, %ub, %step_dynamic = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
%step_static = rv32.li 4 : !riscv.reg
riscv_scf.for %i_static : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_static : !riscv.reg {
  riscv_scf.yield
}
// CHECK:    riscv_scf.for %i_static : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step 4 : si12 {


// Nothing to be done
riscv_scf.for %i_dynamic : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_dynamic : !riscv.reg {
  riscv_scf.yield
}

// CHECK:    riscv_scf.for %i_dynamic : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_dynamic : !riscv.reg {


// Only support rv32.LiOp for now
%step_static_rv64 = rv64.li 4 : !riscv.reg
riscv_scf.for %i_rv64 : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_static_rv64 : !riscv.reg {
  riscv_scf.yield
}

// CHECK:    riscv_scf.for %i_rv64 : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_static_rv64 : !riscv.reg {


// Cannot support labels for now
%step_static_label = rv64.li "hello" : !riscv.reg
riscv_scf.for %i_label : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_static_label : !riscv.reg {
  riscv_scf.yield
}

// CHECK:    riscv_scf.for %i_label : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_static_label : !riscv.reg {


// Value too large
%step_big = rv32.li 4000 : !riscv.reg
riscv_scf.for %i_big : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_big : !riscv.reg {
  riscv_scf.yield
}

// CHECK:    riscv_scf.for %i_big : !riscv.reg = %lb : !riscv.reg to %ub : !riscv.reg step %step_big : !riscv.reg {
