// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

riscv_func.func @xfrep() {
  %0 = riscv.get_register : !riscv.reg
  %1 = riscv.get_register : !riscv.reg

  // RISC-V extensions
  riscv_snitch.scfgw %0, %1 : (!riscv.reg, !riscv.reg) -> ()
  // CHECK: riscv_snitch.scfgw %0, %1 : (!riscv.reg, !riscv.reg) -> ()
  riscv_snitch.scfgwi %0, 42 : (!riscv.reg) -> ()
  // CHECK-NEXT: riscv_snitch.scfgwi %0, 42 : (!riscv.reg) -> ()

  riscv_snitch.frep_outer %0 {
    %add_o = riscv.add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
  }
  // CHECK-NEXT:  riscv_snitch.frep_outer %0 {
  // CHECK-NEXT:    %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
  // CHECK-NEXT:  }

  riscv_snitch.frep_inner %0 {
    %add_i = riscv.add %0, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
  }
  // CHECK-NEXT:  riscv_snitch.frep_inner %0 {
  // CHECK-NEXT:    %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
  // CHECK-NEXT:  }

  %readable = riscv_snitch.get_stream : !stream.readable<!riscv.freg<ft0>>
  %writable = riscv_snitch.get_stream : !stream.writable<!riscv.freg<ft1>>
  riscv_snitch.frep_outer %0 {
    %val0 = riscv_snitch.read from %readable : !riscv.freg<ft0>
    %val1 = riscv.fmv.d %val0 : (!riscv.freg<ft0>) -> !riscv.freg<ft1>
    riscv_snitch.write %val1 to %writable : !riscv.freg<ft1>
  }
  // CHECK-NEXT:  %readable = riscv_snitch.get_stream : !stream.readable<!riscv.freg<ft0>>
  // CHECK-NEXT:  %writable = riscv_snitch.get_stream : !stream.writable<!riscv.freg<ft1>>
  // CHECK-NEXT:  riscv_snitch.frep_outer %0 {
  // CHECK-NEXT:    %val0 = riscv_snitch.read from %readable : !riscv.freg<ft0>
  // CHECK-NEXT:    %val1 = riscv.fmv.d %val0 : (!riscv.freg<ft0>) -> !riscv.freg<ft1>
  // CHECK-NEXT:    riscv_snitch.write %val1 to %writable : !riscv.freg<ft1>
  // CHECK-NEXT:  }

  %init = "test.op"() : () -> (!riscv.freg<ft3>)
  %z = riscv_snitch.frep_outer %0 iter_args(%acc = %init) -> (!riscv.freg<ft3>) {
    %res = riscv.fadd.d %acc, %acc : (!riscv.freg<ft3>, !riscv.freg<ft3>) -> !riscv.freg<ft3>
    riscv_snitch.frep_yield %res : !riscv.freg<ft3>
  }

  // CHECK-NEXT:  %init = "test.op"() : () -> !riscv.freg<ft3>
  // CHECK-NEXT:    %z = riscv_snitch.frep_outer %0 iter_args(%acc = %init) -> (!riscv.freg<ft3>) {
  // CHECK-NEXT:      %res = riscv.fadd.d %acc, %acc : (!riscv.freg<ft3>, !riscv.freg<ft3>) -> !riscv.freg<ft3>
  // CHECK-NEXT:      riscv_snitch.frep_yield %res : !riscv.freg<ft3>
  // CHECK-NEXT:    }

  // Terminate block
  riscv_func.return
}

riscv_func.func @xdma() {
  %reg = riscv.get_register : !riscv.reg
  // CHECK: %reg = riscv.get_register : !riscv.reg


  riscv_snitch.dmsrc %reg, %reg : (!riscv.reg, !riscv.reg) -> ()
  // CHECK-NEXT: riscv_snitch.dmsrc %reg, %reg : (!riscv.reg, !riscv.reg) -> ()

  riscv_snitch.dmdst %reg, %reg : (!riscv.reg, !riscv.reg) -> ()
  // CHECK-NEXT: riscv_snitch.dmdst %reg, %reg : (!riscv.reg, !riscv.reg) -> ()

  riscv_snitch.dmstr %reg, %reg : (!riscv.reg, !riscv.reg) -> ()
  // CHECK-NEXT: riscv_snitch.dmstr %reg, %reg : (!riscv.reg, !riscv.reg) -> ()
  riscv_snitch.dmrep %reg : (!riscv.reg) -> ()
  // CHECK-NEXT: riscv_snitch.dmrep %reg : (!riscv.reg) -> ()

  %0 = riscv_snitch.dmcpy %reg, %reg : (!riscv.reg, !riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{\d+}} = riscv_snitch.dmcpy %reg, %reg : (!riscv.reg, !riscv.reg) -> !riscv.reg
  %1 = riscv_snitch.dmstat %reg : (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{\d+}} = riscv_snitch.dmstat %reg : (!riscv.reg) -> !riscv.reg

  %2 = riscv_snitch.dmcpyi %reg, 0 : (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT: %{{\d+}} = riscv_snitch.dmcpyi %reg, 0 : (!riscv.reg) -> !riscv.reg
  %3 = riscv_snitch.dmstati 0 : () -> !riscv.reg
  // CHECK-NEXT: %{{\d+}} = riscv_snitch.dmstati 0 : () -> !riscv.reg


  riscv_func.return
}

riscv_func.func @simd() {
  %v = riscv.get_float_register : !riscv.freg
  // CHECK: %v = riscv.get_float_register : !riscv.freg

  %0 = riscv_snitch.vfmul.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg
  // CHECK-NEXT: %0 = riscv_snitch.vfmul.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg

  %1 = riscv_snitch.vfadd.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg
  // CHECK-NEXT: %1 = riscv_snitch.vfadd.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg

  %2 = riscv_snitch.vfcpka.s.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg
  // CHECK-NEXT: %2 = riscv_snitch.vfcpka.s.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg

  %3 = riscv_snitch.vfmac.s %v, %v, %v : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
  // CHECK-NEXT: %3 = riscv_snitch.vfmac.s %v, %v, %v : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg

  %4 = riscv_snitch.vfsum.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg
  // CHECK-NEXT: %4 = riscv_snitch.vfsum.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg

  %5 = riscv_snitch.vfadd.h %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg
  // CHECK-NEXT: %5 = riscv_snitch.vfadd.h %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg

  %6 = riscv_snitch.vfmax.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg
  // CHECK-NEXT: %6 = riscv_snitch.vfmax.s %v, %v : (!riscv.freg, !riscv.freg) -> !riscv.freg

  riscv_func.return
}


// CHECK-GENERIC-NEXT: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:     %0 = "riscv.get_register"() : () -> !riscv.reg
// CHECK-GENERIC-NEXT:     %1 = "riscv.get_register"() : () -> !riscv.reg
// CHECK-GENERIC-NEXT:     "riscv_snitch.scfgw"(%0, %1) : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:     "riscv_snitch.scfgwi"(%0) {"immediate" = 42 : si12} : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:    "riscv_snitch.frep_outer"(%{{.*}}) ({
// CHECK-GENERIC-NEXT:      %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv_snitch.frep_yield"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:    "riscv_snitch.frep_inner"(%{{.*}}) ({
// CHECK-GENERIC-NEXT:      %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:      "riscv_snitch.frep_yield"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:        %readable = "riscv_snitch.get_stream"() : () -> !stream.readable<!riscv.freg<ft0>>
// CHECK-GENERIC-NEXT:        %writable = "riscv_snitch.get_stream"() : () -> !stream.writable<!riscv.freg<ft1>>
// CHECK-GENERIC-NEXT:        "riscv_snitch.frep_outer"(%0) ({
// CHECK-GENERIC-NEXT:          %val0 = "riscv_snitch.read"(%readable) : (!stream.readable<!riscv.freg<ft0>>) -> !riscv.freg<ft0>
// CHECK-GENERIC-NEXT:          %val1 = "riscv.fmv.d"(%val0) : (!riscv.freg<ft0>) -> !riscv.freg<ft1>
// CHECK-GENERIC-NEXT:          "riscv_snitch.write"(%val1, %writable) : (!riscv.freg<ft1>, !stream.writable<!riscv.freg<ft1>>) -> ()
// CHECK-GENERIC-NEXT:          "riscv_snitch.frep_yield"() : () -> ()
// CHECK-GENERIC-NEXT:        }) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:    %init = "test.op"() : () -> !riscv.freg<ft3>
// CHECK-GENERIC-NEXT:    %z = "riscv_snitch.frep_outer"(%0, %init) ({
// CHECK-GENERIC-NEXT:    ^0(%acc : !riscv.freg<ft3>):
// CHECK-GENERIC-NEXT:      %res = "riscv.fadd.d"(%acc, %acc) {"fastmath" = #riscv.fastmath<none>} : (!riscv.freg<ft3>, !riscv.freg<ft3>) -> !riscv.freg<ft3>
// CHECK-GENERIC-NEXT:      "riscv_snitch.frep_yield"(%res) : (!riscv.freg<ft3>) -> ()
// CHECK-GENERIC-NEXT:    }) {"stagger_mask" = #builtin.int<0>, "stagger_count" = #builtin.int<0>} : (!riscv.reg, !riscv.freg<ft3>) -> !riscv.freg<ft3>
// CHECK-GENERIC-NEXT:     "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) {"sym_name" = "xfrep", "function_type" = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:   "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:     %reg = "riscv.get_register"() : () -> !riscv.reg
// CHECK-GENERIC-NEXT:     "riscv_snitch.dmsrc"(%reg, %reg) : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:     "riscv_snitch.dmdst"(%reg, %reg) : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:     "riscv_snitch.dmstr"(%reg, %reg) : (!riscv.reg, !riscv.reg) -> ()
// CHECK-GENERIC-NEXT:     "riscv_snitch.dmrep"(%reg) : (!riscv.reg) -> ()
// CHECK-GENERIC-NEXT:     %{{.*}} = "riscv_snitch.dmcpy"(%reg, %reg) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:     %{{.*}} = "riscv_snitch.dmstat"(%reg) : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:     %{{.*}} = "riscv_snitch.dmcpyi"(%reg) <{"config" = 0 : ui5}> : (!riscv.reg) -> !riscv.reg
// CHECK-GENERIC-NEXT:     %{{.*}} = "riscv_snitch.dmstati"() <{"status" = 0 : ui5}> : () -> !riscv.reg
// CHECK-GENERIC-NEXT:     "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) {"sym_name" = "xdma", "function_type" = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:   "riscv_func.func"() ({
// CHECK-GENERIC-NEXT:       %v = "riscv.get_float_register"() : () -> !riscv.freg
// CHECK-GENERIC-NEXT:       %0 = "riscv_snitch.vfmul.s"(%v, %v) {"fastmath" = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:       %1 = "riscv_snitch.vfadd.s"(%v, %v) {"fastmath" = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:       %2 = "riscv_snitch.vfcpka.s.s"(%v, %v) : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:       %3 = "riscv_snitch.vfmac.s"(%v, %v, %v) {"fastmath" = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:       %4 = "riscv_snitch.vfsum.s"(%v, %v) : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:       %5 = "riscv_snitch.vfadd.h"(%v, %v) {"fastmath" = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:       %6 = "riscv_snitch.vfmax.s"(%v, %v) {"fastmath" = #riscv.fastmath<none>} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-GENERIC-NEXT:       "riscv_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) {"sym_name" = "simd", "function_type" = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
