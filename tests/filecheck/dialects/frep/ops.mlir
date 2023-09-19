// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%i0 = "test.op"() : () -> !riscv.reg<a0>
%ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
riscv.frep_outer %i0, 0, 0 ({
    %ft2 = riscv.vfadd.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
}) : (!riscv.reg<a0>) -> ()
riscv.frep_inner %i0, 0, 0 ({
    %ft2 = riscv.vfadd.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
}) : (!riscv.reg<a0>) -> ()

// CHECK:       %i0 = "test.op"() : () -> !riscv.reg<a0>
// CHECK-NEXT:  %ft0, %ft1 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
// CHECK-NEXT:  riscv.frep_outer %i0, 0, 0 ({
// CHECK-NEXT:      %{{.*}} = riscv.vfadd.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-NEXT:  }) : (!riscv.reg<a0>) -> ()
// CHECK-NEXT:  riscv.frep_inner %i0, 0, 0 ({
// CHECK-NEXT:      %{{.*}} = riscv.vfadd.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-NEXT:  }) : (!riscv.reg<a0>) -> ()

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %{{.*}} = "test.op"() : () -> !riscv.reg<a0>
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft1>)
// CHECK-GENERIC-NEXT:    "riscv.frep_outer"(%{{.*}}) ({
// CHECK-GENERIC-NEXT:      %{{.*}} = "riscv.vfadd.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-GENERIC-NEXT:    }) {"stagger_mask" = #int<0>, "stagger_count" = #int<0>} : (!riscv.reg<a0>) -> ()
// CHECK-GENERIC-NEXT:    "riscv.frep_inner"(%{{.*}}) ({
// CHECK-GENERIC-NEXT:      %{{.*}} = "riscv.vfadd.s"(%{{.*}}, %{{.*}}) : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
// CHECK-GENERIC-NEXT:    }) {"stagger_mask" = #int<0>, "stagger_count" = #int<0>} : (!riscv.reg<a0>) -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()

