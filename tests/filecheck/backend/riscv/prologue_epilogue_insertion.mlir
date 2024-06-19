// RUN: xdsl-opt --split-input-file -p "riscv-prologue-epilogue-insertion" %s | filecheck %s
// RUN: xdsl-opt --split-input-file -p "riscv-prologue-epilogue-insertion{flen=4}" %s | filecheck %s --check-prefix=SMALL_FLEN

// CHECK: func @main
riscv_func.func @main() {
  // CHECK-NEXT: get_register
  // CHECK-SAME: -> !riscv.reg<sp>
  // CHECK-NEXT: addi %{{.*}}, -12
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.reg<sp>
  // CHECK-NEXT: get_float_register
  // CHECK-SAME: -> !riscv.freg<fs2>
  // CHECK-NEXT: fsd %{{.*}}, %{{.*}}, 0
  // CHECK-SAME: (!riscv.reg<sp>, !riscv.freg<fs2>) -> ()
  // CHECK-NEXT: get_register
  // CHECK-SAME: -> !riscv.reg<s5>
  // CHECK-NEXT: sw %{{.*}}, %{{.*}}, 8
  // CHECK-SAME: (!riscv.reg<sp>, !riscv.reg<s5>) -> ()

  %fs0 = riscv.get_float_register : () -> !riscv.freg<fs0>
  %fs1 = riscv.get_float_register : () -> !riscv.freg<fs1>
  // Clobber only fs2.
  %sum1 = riscv.fadd.s %fs0, %fs1 : (!riscv.freg<fs0>, !riscv.freg<fs1>) -> !riscv.freg<fs2>
  %zero = riscv.get_register : () -> !riscv.reg<zero>
  // Clobber s5.
  %0 = riscv.mv %zero : (!riscv.reg<zero>) -> !riscv.reg<s5>
  riscv_cf.blt %0 : !riscv.reg<s5>, %zero : !riscv.reg<zero>, ^0(), ^1()
^1:
  // CHECK: label "l1"
  riscv.label "l1"
  // CHECK-NEXT: fld %{{.*}}, 0
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.freg<fs2>
  // CHECK-NEXT: lw %{{.*}}, 8
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.freg<s5>
  // CHECK-NEXT: addi %{{.*}}, 12
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.freg<sp>
  // CHECK-NEXT: return
  riscv_func.return
^0:
  // CHECK: label "l0"
  riscv.label "l0"
  // CHECK-NEXT: fld %{{.*}}, 0
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.freg<fs2>
  // CHECK-NEXT: lw %{{.*}}, 8
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.freg<s5>
  // CHECK-NEXT: addi %{{.*}}, 12
  // CHECK-SAME: (!riscv.reg<sp>) -> !riscv.freg<sp>
  riscv_func.return
}

// SMALL_FLEN: func @main
// SMALL_FLEN: addi %{{.*}}, -8
// SMALL_FLEN-SAME: (!riscv.reg<sp>) -> !riscv.reg<sp>