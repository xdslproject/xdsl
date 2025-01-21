// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else(%e0 : !riscv.reg<a2>, %e1 : !riscv.reg<a3>):
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
  ^then(%t0 : !riscv.reg<a2>, %t1 : !riscv.reg<a3>):
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
}) : () -> ()

// CHECK: Operation does not verify: riscv_cf branch op then block first op must be a label

// -----

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else(%4 : !riscv.reg<a2>, %4 : !riscv.reg<a2>)
  ^else(%e0 : !riscv.reg<a2>, %e1 : !riscv.reg<a3>):
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
  ^then(%t0 : !riscv.reg<a2>, %t1 : !riscv.reg<a3>):
    riscv.label "label"
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
}) : () -> ()

// CHECK: Operation does not verify: Block arg types must match !riscv.reg<a2> !riscv.reg<a3>

// -----

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%3 : !riscv.reg<a3>, %3 : !riscv.reg<a3>), ^else(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else(%e0 : !riscv.reg<a2>, %e1 : !riscv.reg<a3>):
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
  ^then(%t0 : !riscv.reg<a2>, %t1 : !riscv.reg<a3>):
    riscv.label "label"
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
}) : () -> ()

// CHECK: Operation does not verify: Block arg types must match !riscv.reg<a3> !riscv.reg<a2>

// -----

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^then(%t0 : !riscv.reg<a2>, %t1 : !riscv.reg<a3>):
    riscv.label "label"
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
  ^else(%e0 : !riscv.reg<a2>, %e1 : !riscv.reg<a3>):
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
}) : () -> ()

// CHECK: Operation does not verify: riscv_cf branch op else block must be immediately after op

// -----

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.branch ^0(%0 : !riscv.reg<a0>, %0 : !riscv.reg<a0>)
  ^0(%t0 : !riscv.reg<a2>, %t1 : !riscv.reg<a3>):
    riscv_cf.j ^0(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
}) : () -> ()

// CHECK: Operation does not verify: Block arg types must match !riscv.reg<a0> !riscv.reg<a2>

// -----

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.branch ^1(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>)
  ^0(%t0 : !riscv.reg<a0>, %t1 : !riscv.reg<a1>):
    riscv_cf.j ^0(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
  ^1(%t2 : !riscv.reg<a0>, %t3 : !riscv.reg<a1>):
    riscv_cf.j ^0(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
}) : () -> ()

// CHECK: Operation does not verify: Successor block must be immediately after parent block in the parent region.

// -----

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.j ^0(%1 : !riscv.reg<a1>, %1 : !riscv.reg<a1>)
  ^0(%t0 : !riscv.reg<a0>, %t1 : !riscv.reg<a1>):
    riscv.label "label"
    riscv_cf.j ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>)
}) : () -> ()

// CHECK: Operation does not verify: Block arg types must match !riscv.reg<a1> !riscv.reg<a0>

// -----

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.branch ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>)
  ^0(%t0 : !riscv.reg<a0>, %t1 : !riscv.reg<a1>):
    riscv_cf.j ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>)
}) : () -> ()

// CHECK: Operation does not verify: riscv_cf.j operation successor must have a riscv.label operation as a first argument, found JOp(riscv_cf.j ^0(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>))
