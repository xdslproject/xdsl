// RUN: xdsl-opt -t riscv-asm %s | filecheck %s


riscv_func.func @main() {
  %0 = riscv.get_register : () -> !riscv.reg<a0>
  %1 = riscv.get_register : () -> !riscv.reg<a1>

  %readable = riscv_snitch.get_stream : !stream.readable<!riscv.freg<ft0>>
  %writable = riscv_snitch.get_stream : !stream.writable<!riscv.freg<ft1>>
  riscv_snitch.frep_outer %0 {
    %val0 = riscv_snitch.read from %readable : !riscv.freg<ft0>
    %val1 = riscv.fmv.d %val0 : (!riscv.freg<ft0>) -> !riscv.freg<ft1>
    riscv_snitch.write %val1 to %writable : !riscv.freg<ft1>
  }

  riscv_snitch.dmsrc %0, %1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> ()
  riscv_snitch.dmdst %0, %1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> ()
  %2 = riscv_snitch.dmcpyi %0, 0 : (!riscv.reg<a0>) -> !riscv.reg<a2>
  riscv_snitch.dmstr %0, %1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> ()
  riscv_snitch.dmrep %0 : (!riscv.reg<a0>) -> ()
  %3 = riscv_snitch.dmcpy %0, %2 : (!riscv.reg<a0>, !riscv.reg<a2>) -> !riscv.reg<a3>
  %4 = riscv_snitch.dmstat %3 : (!riscv.reg<a3>) -> !riscv.reg<a4>
  %5 = riscv_snitch.dmstati 22 : () -> !riscv.reg<a5>

  riscv_func.return
}

// CHECK:        main:
// CHECK-NEXT:       frep.o a0, 1, 0, 0
// CHECK-NEXT:       fmv.d ft1, ft0
// CHECK-NEXT:       dmsrc a0, a1
// CHECK-NEXT:       dmdst a0, a1
// CHECK-NEXT:       dmcpyi a2, a0, 0
// CHECK-NEXT:       dmstr a0, a1
// CHECK-NEXT:       dmrep a0
// CHECK-NEXT:       dmcpy a3, a0, a2
// CHECK-NEXT:       dmstat a4, a3
// CHECK-NEXT:       dmstati a5, 22
// CHECK-NEXT:       ret
