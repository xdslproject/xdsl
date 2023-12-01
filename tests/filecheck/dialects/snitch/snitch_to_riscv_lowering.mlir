// RUN: xdsl-opt -p lower-snitch %s | filecheck %s
builtin.module {
  %addr = "test.op"() : () -> !riscv.reg<>
  %bound = riscv.li 9 : () -> !riscv.reg<>
  %stride = riscv.li 4 : () -> !riscv.reg<>
  %rep = riscv.li 0 : () -> !riscv.reg<>
  // SSR setup sequence for dimension 0
  "snitch.ssr_set_dimension_bound"(%bound) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 64 : () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv_snitch.scfgw %bound, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_stride"(%stride) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 192 : () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} riscv_snitch.scfgw %stride, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_source"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 768 : () -> !riscv.reg<>
  // %{{.*}} = riscv_snitch.scfgw %addr, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_destination"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 896 : () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv_snitch.scfgw %addr, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  // SSR setup sequence for dimension 3
  "snitch.ssr_set_dimension_bound"(%bound) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 160 : () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv_snitch.scfgw %bound, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_stride"(%stride) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 288 : () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv_snitch.scfgw %stride, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_source"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 864 : () -> !riscv.reg<>
  // riscv_snitch.scfgw %addr, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_destination"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<3>} : (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 992 : () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv_snitch.scfgw %addr, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_stream_repetition"(%rep) {"dm" = #builtin.int<0>}: (!riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.li 32 : () -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv_snitch.scfgw %rep, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  // On/Off switching sequence
  "snitch.ssr_enable"() : () -> ()
  // CHECK: riscv.csrrsi 1984, 1
  "snitch.ssr_disable"() : () -> ()
  // CHECK: riscv.csrrci 1984, 1
}
