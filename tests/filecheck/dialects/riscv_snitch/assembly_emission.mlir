// RUN: xdsl-opt -t riscv-asm %s | filecheck %s


riscv_func.func @main() {
  %0 = rv32.get_register : !riscv.reg<a0>
  %1 = rv32.get_register : !riscv.reg<a1>

  %readable = riscv_snitch.get_stream : !snitch.readable<!riscv.freg<ft0>>
  %writable = riscv_snitch.get_stream : !snitch.writable<!riscv.freg<ft1>>
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

  %ft0 = riscv.get_float_register : !riscv.freg<ft0>
  %ft1 = riscv.get_float_register : !riscv.freg<ft1>
  %ft2 = riscv.get_float_register : !riscv.freg<ft2>
  %ft3 = riscv.get_float_register : !riscv.freg<ft3>

  // f32
  %vfmul_s = riscv_snitch.vfmul.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
  %vfadd_s = riscv_snitch.vfadd.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
  %vfmax_s = riscv_snitch.vfmax.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
  %vfcpka_s_s = riscv_snitch.vfcpka.s.s %ft0, %ft1 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
  %vfmac_s = riscv_snitch.vfmac.s %ft3, %ft0, %ft1 : (!riscv.freg<ft3>, !riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft3>
  %vfsum_s = riscv_snitch.vfsum.s %vfmac_s, %ft1 : (!riscv.freg<ft3>, !riscv.freg<ft1>) -> !riscv.freg<ft3>

  // f16
  %vfadd_h = riscv_snitch.vfadd.h %ft1, %ft0 : (!riscv.freg<ft1>, !riscv.freg<ft0>) -> !riscv.freg<ft2>

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
// CHECK-NEXT:       vfmul.s ft2, ft0, ft1
// CHECK-NEXT:       vfadd.s ft2, ft0, ft1
// CHECK-NEXT:       vfmax.s ft2, ft0, ft1
// CHECK-NEXT:       vfcpka.s.s ft2, ft0, ft1
// CHECK-NEXT:       vfmac.s ft3, ft0, ft1
// CHECK-NEXT:       vfsum.s ft3, ft1
// CHECK-NEXT:       vfadd.h ft2, ft1, ft0
// CHECK-NEXT:       ret
