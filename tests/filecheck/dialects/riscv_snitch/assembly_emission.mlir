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

  riscv_snitch.dmstr %0, %1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> ()
  riscv_snitch.dmrep %0 : (!riscv.reg<a0>) -> ()

  %a = riscv_snitch.dmcpy %0, %1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<zero>
  %b = riscv_snitch.dmstat %0 : (!riscv.reg<a0>) -> !riscv.reg<zero>

  %c = riscv_snitch.dmcpyi %0, 0 : (!riscv.reg<a0>) -> !riscv.reg<a2>

  %d = riscv_snitch.dmstati 0 : () -> !riscv.reg<a2>

  riscv_func.return
}

// CHECK-NEXT: main:
// CHECK-NEXT:       frep.o a0, 1, 0, 0
// CHECK-NEXT:       fmv.d ft1, ft0
// CHECK-NEXT:       dmsrc a0, a1
// CHECK-NEXT:       dmdst a0, a1
// CHECK-NEXT:       dmstr a0, a1
// CHECK-NEXT:       dmrep a0
// CHECK-NEXT:       dmcpy zero, a0, a1
// CHECK-NEXT:       dmstat zero, a0
// CHECK-NEXT:       dmcpyi a2, a0, 0
// CHECK-NEXT:       dmstati a2, 0
// CHECK-NEXT:       ret
