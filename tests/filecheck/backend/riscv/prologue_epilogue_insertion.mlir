// RUN: xdsl-opt --split-input-file -p "riscv-prologue-epilogue-insertion" %s | filecheck %s
// RUN: xdsl-opt --split-input-file -p "riscv-prologue-epilogue-insertion{flen=4}" %s | filecheck %s --check-prefix=CHECK-SMALL-FLEN

// CHECK: func @main
riscv_func.func @main() {
  // CHECK-NEXT: get_register
  // CHECK-SAME: : !riscv.reg<sp>
  // CHECK-NEXT: addi %{{.*}}, -12
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.reg<sp>
  // CHECK-NEXT: get_float_register
  // CHECK-SAME: : !riscv.freg<fs2>
  // CHECK-NEXT: fsd %{{.*}}, %{{.*}}, 0
  // CHECK-SAME: (!riscv.reg<sp>, !riscv.freg<fs2>) -> ()
  // CHECK-NEXT: get_register
  // CHECK-SAME: : !riscv.reg<s5>
  // CHECK-NEXT: sw %{{.*}}, %{{.*}}, 8
  // CHECK-SAME: (!riscv.reg<sp>, !riscv.reg<s5>) -> ()

  %fs0 = riscv.get_float_register : !riscv.freg<fs0>
  %fs1 = riscv.get_float_register : !riscv.freg<fs1>
  // Clobber only fs2.
  %sum1 = riscv.fadd.s %fs0, %fs1 : (!riscv.freg<fs0>, !riscv.freg<fs1>) -> !riscv.freg<fs2>
  %zero = riscv.get_register : !riscv.reg<zero>
  // Clobber s5.
  %0 = riscv.mv %zero : (!riscv.reg<zero>) -> !riscv.reg<s5>
  riscv_cf.blt %0 : !riscv.reg<s5>, %zero : !riscv.reg<zero>, ^bb0(), ^bb1()
^bb1:
  // CHECK: label "l1"
  riscv.label "l1"
  // CHECK-NEXT: fld %{{.*}}, 0
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.freg<fs2>
  // CHECK-NEXT: lw %{{.*}}, 8
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.reg<s5>
  // CHECK-NEXT: addi %{{.*}}, 12
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.reg<sp>
  // CHECK-NEXT: return
  riscv_func.return
^bb0:
  // CHECK: label "l0"
  riscv.label "l0"
  // CHECK-NEXT: fld %{{.*}}, 0
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.freg<fs2>
  // CHECK-NEXT: lw %{{.*}}, 8
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.reg<s5>
  // CHECK-NEXT: addi %{{.*}}, 12
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.reg<sp>
  riscv_func.return
}

// CHECK-SMALL-FLEN: func @main
// CHECK-SMALL-FLEN: addi %{{.*}}, -8
// CHECK-SMALL-FLEN-SAME: (!riscv.reg<sp>) -> !riscv.reg<sp>

// CHECK: func @simple
riscv_func.func @simple(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) -> !riscv.reg<a0> {
  // CHECK-NOT: %{{.*}} = riscv.get_register : !riscv.reg<sp>
  // CHECK-NOT: %{{.*}} = riscv.addi %{{.*}}, 0 : (!riscv.reg<sp>) -> !riscv.reg<sp>

  // CHECK-NEXT: %{{.*}} = riscv.mv %{{\S+}}
  // CHECK-SAME: (!riscv.reg<a0>) -> !riscv.reg<t0>
  %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<t0>
  %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<t2>
  %4 = riscv.li 10 : !riscv.reg<t1>
  %5 = riscv.add %2, %3 : (!riscv.reg<t0>, !riscv.reg<t2>) -> !riscv.reg<t0>
  %6 = riscv.add %5, %4 : (!riscv.reg<t0>, !riscv.reg<t1>) -> !riscv.reg<t0>
  %7 = riscv.mv %6 : (!riscv.reg<t0>) -> !riscv.reg<a0>

  // CHECK-NOT: %{{.*}} = riscv.addi %{{.*}}, 0 : (!riscv.reg<sp>) -> !riscv.reg<sp>

  // CHECK: riscv_func.return %{{\S+}}
  // CHECK-SAME: !riscv.reg<a0>
  riscv_func.return %7 : !riscv.reg<a0>
}

// CHECK: func @simplefp
riscv_func.func @simplefp(%0 : !riscv.freg<fa0>, %1 : !riscv.freg<fa1>) -> !riscv.freg<fa0> {
  // CHECK-NOT: %{{.*}} = riscv.get_register : !riscv.reg<sp>
  // CHECK-NOT: %{{.*}} = riscv.addi %{{.*}}, 0 : (!riscv.reg<sp>) -> !riscv.reg<sp>

  // CHECK-NEXT: %{{.*}} = riscv.fmv.s %{{\S+}}
  // CHECK-SAME: : (!riscv.freg<fa0>) -> !riscv.freg<ft0>
  %2 = riscv.fmv.s %0 : (!riscv.freg<fa0>) -> !riscv.freg<ft0>
  %3 = riscv.fmv.s %1 : (!riscv.freg<fa1>) -> !riscv.freg<ft1>
  %4 = riscv.fadd.s %2, %3 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
  %5 = riscv.fmv.s %4 : (!riscv.freg<ft0>) -> !riscv.freg<fa0>

  // CHECK-NOT: %{{.*}} = riscv.addi %{{.*}}, 0 : (!riscv.reg<sp>) -> !riscv.reg<sp>

  // CHECK: riscv_func.return %{{\S+}}
  // CHECK-SAME: !riscv.freg<fa0>
  riscv_func.return %5 : !riscv.freg<fa0>
}
