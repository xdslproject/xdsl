// RUN: xdsl-opt -t riscv-asm %s | filecheck %s


riscv_func.func @main() {
  %0 = riscv.get_register : !riscv.reg<a0>
  %1 = riscv.get_register : !riscv.reg<a1>

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

  %f0 = riscv.get_float_register : !riscv.freg<ft0>
  %f1 = riscv_snitch.vfmul.s %f0, %f0 : (!riscv.freg<ft0>, !riscv.freg<ft0>) -> !riscv.freg<ft1>
  %f2 = riscv_snitch.vfadd.s %f0, %f0 : (!riscv.freg<ft0>, !riscv.freg<ft0>) -> !riscv.freg<ft1>
  %f3 = riscv_snitch.vfcpka.s.s %f0, %f0 : (!riscv.freg<ft0>, !riscv.freg<ft0>) -> !riscv.freg<ft1>
  %f4 = riscv_snitch.vfmac.s %f3, %f0, %f0 : (!riscv.freg<ft1>, !riscv.freg<ft0>, !riscv.freg<ft0>) -> !riscv.freg<ft1>
  %f5 = riscv_snitch.vfsum.s %f4, %f0 : (!riscv.freg<ft1>, !riscv.freg<ft0>) -> !riscv.freg<ft1>

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
// CHECK-NEXT:       vfmul.s ft1, ft0, ft0
// CHECK-NEXT:       vfadd.s ft1, ft0, ft0
// CHECK-NEXT:       vfcpka.s.s ft1, ft0, ft0
// CHECK-NEXT:       vfmac.s ft1, ft0, ft0
// CHECK-NEXT:       vfsum.s ft1, ft0
// CHECK-NEXT:       ret
