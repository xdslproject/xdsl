// RUN: xdsl-opt -t riscv-asm %s | filecheck %s

%0 = riscv.get_register : () -> !riscv.reg<a0>
%1 = riscv.get_register : () -> !riscv.reg<a1>
%2 = riscv.get_register : () -> !riscv.reg<a2>
%3 = riscv.get_register : () -> !riscv.reg<a3>
%4 = riscv.get_register : () -> !riscv.reg<a2>
%5 = riscv.get_register : () -> !riscv.reg<a3>

riscv_func.func @main() {
    riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else0(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else0(%e00 : !riscv.reg<a2>, %e01 : !riscv.reg<a3>):
    riscv.label "else0"
    riscv_cf.bne %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else1(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else1(%e10 : !riscv.reg<a2>, %e11 : !riscv.reg<a3>):
    riscv.label "else1"
    riscv_cf.blt %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else2(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else2(%e20 : !riscv.reg<a2>, %e21 : !riscv.reg<a3>):
    riscv.label "else2"
    riscv_cf.bge %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else3(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else3(%e30 : !riscv.reg<a2>, %e31 : !riscv.reg<a3>):
    riscv.label "else3"
    riscv_cf.bltu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else4(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else4(%e40 : !riscv.reg<a2>, %e41 : !riscv.reg<a3>):
    riscv.label "else4"
    riscv_cf.bgeu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else5(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else5(%e50 : !riscv.reg<a2>, %e51 : !riscv.reg<a3>):
    riscv.label "else5"
    riscv_cf.branch ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
  ^then(%t0 : !riscv.reg<a2>, %t1 : !riscv.reg<a3>):
    riscv.label "then"
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>)
}

// CHECK:       main:
// CHECK-NEXT:      beq a0, a1, then
// CHECK-NEXT:  else0:
// CHECK-NEXT:      bne a0, a1, then
// CHECK-NEXT:  else1:
// CHECK-NEXT:      blt a0, a1, then
// CHECK-NEXT:  else2:
// CHECK-NEXT:      bge a0, a1, then
// CHECK-NEXT:  else3:
// CHECK-NEXT:      bltu a0, a1, then
// CHECK-NEXT:  else4:
// CHECK-NEXT:      bgeu a0, a1, then
// CHECK-NEXT:  else5:
// CHECK-NEXT:  then:
// CHECK-NEXT:      j then
