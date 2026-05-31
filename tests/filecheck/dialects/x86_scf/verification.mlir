// RUN: xdsl-opt --parsing-diagnostics --verify-diagnostics --split-input-file %s | filecheck %s

%lb, %ub, %step = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64)

"x86_scf.for"(%lb, %ub, %step) <{ub_attr = 10 : si32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
^bb0(%i: !x86.reg64):
    x86_scf.yield
}) : (!x86.reg64, !x86.reg64, !x86.reg64) -> ()

// CHECK: Operation does not verify: Exactly one of ub_attr (static) or ub_val (dynamic) must be set

// -----

%lb, %ub, %step = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64)

"x86_scf.for"(%lb, %ub, %step) <{step_attr = 1 : si32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
^bb0(%i: !x86.reg64):
    x86_scf.yield
}) : (!x86.reg64, !x86.reg64, !x86.reg64) -> ()

// CHECK: Operation does not verify: Exactly one of step_attr (static) or step_val (dynamic) must be set

// -----

%lb = "test.op"() : () -> !x86.reg64

x86_scf.for %i : !x86.reg64 = %lb to 1 : f32 step 1 : si32 {
    x86_scf.yield
}

// CHECK: Expected IntegerAttr

// -----

%lb = "test.op"() : () -> !x86.reg64

x86_scf.for %i : !x86.reg64 = %lb to 1 : si32 step 1 : f32 {
    x86_scf.yield
}

// CHECK: Expected IntegerAttr

// -----

%lb = "test.op"() : () -> !x86.reg64

x86_scf.for %i : !x86.reg64 = %lb to 1 : i64 step 1 : si32 {
    x86_scf.yield
}

// CHECK: Expected attribute si32 but got i64

// -----

%lb = "test.op"() : () -> !x86.reg64

x86_scf.for %i : !x86.reg64 = %lb to 1 : si32 step 1 : i64 {
    x86_scf.yield
}

// CHECK: Expected attribute si32 but got i64
