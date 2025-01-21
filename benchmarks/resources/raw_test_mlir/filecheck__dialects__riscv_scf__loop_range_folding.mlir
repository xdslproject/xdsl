// RUN: xdsl-opt -p riscv-scf-loop-range-folding %s | filecheck %s

// CHECK:       builtin.module {

%lb, %ub, %step = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
%0, %1 = "test.op"() : () -> (!riscv.reg, !riscv.reg)
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.reg, !riscv.reg)

riscv_scf.for %2 : !riscv.reg = %lb to %ub step %step {
    // mul by constant
    %3 = riscv.li 3 : !riscv.reg
    %4 = riscv.mul %2, %3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // add with constant
    %5 = riscv.li 5 : !riscv.reg
    %6 = riscv.add %4, %5 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "test.op"(%6) : (!riscv.reg) -> ()
}
// CHECK-NEXT:    %{{.*}} = riscv.li 3 : !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.li 5 : !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      %{{.*}} = riscv.li 3 : !riscv.reg
// CHECK-NEXT:      %{{.*}} = riscv.li 5 : !riscv.reg
// CHECK-NEXT:      "test.op"(%{{.*}}) : (!riscv.reg) -> ()
// CHECK-NEXT:    }

// Don't fold
riscv_scf.for %2 : !riscv.reg = %lb to %ub step %step {
    // Two uses of mul -> can't fold
    %3 = riscv.li 3 : !riscv.reg
    %4 = riscv.mul %2, %3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "test.op"(%4, %2) : (!riscv.reg, !riscv.reg) -> ()
}
riscv_scf.for %2 : !riscv.reg = %lb to %ub step %step {
    // Two uses of add -> can't fold
    %3 = riscv.li 3 : !riscv.reg
    %4 = riscv.add %2, %3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "test.op"(%4, %2) : (!riscv.reg, !riscv.reg) -> ()
}
// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      %{{.*}} = riscv.li 3 : !riscv.reg
// CHECK-NEXT:      %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:      "test.op"(%{{.*}}, %{{.*}}) : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      %{{.*}} = riscv.li 3 : !riscv.reg
// CHECK-NEXT:      %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:      "test.op"(%{{.*}}, %{{.*}}) : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
