// RUN: xdsl-opt -p lower-snitch %s | filecheck %s
builtin.module {
  %addr = "test.op"() : () -> !riscv.reg
  %bound = rv32.li 9 : !riscv.reg
  %stride = rv32.li 4 : !riscv.reg
  %rep = rv32.li 0 : !riscv.reg
  // SSR setup sequence for dimension 0
  "snitch.ssr_set_dimension_bound"(%bound) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 64 : !riscv.reg
  // CHECK-NEXT: riscv_snitch.scfgw %bound, %{{.*}} {comment = "dm 0 dim 0 bound"} : (!riscv.reg, !riscv.reg) -> ()
  "snitch.ssr_set_dimension_stride"(%stride) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 192 : !riscv.reg
  // CHECK-NEXT: riscv_snitch.scfgw %stride, %{{.*}} {comment = "dm 0 dim 0 stride"} : (!riscv.reg, !riscv.reg) -> ()
  "snitch.ssr_set_dimension_source"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 768 : !riscv.reg
  // riscv_snitch.scfgw %addr, %{{.*}} {comment = "dm 0 dim 0 source"} : (!riscv.reg, !riscv.reg) -> ()
  "snitch.ssr_set_dimension_destination"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 896 : !riscv.reg
  // CHECK-NEXT: riscv_snitch.scfgw %addr, %{{.*}} {comment = "dm 0 dim 0 destination"} : (!riscv.reg, !riscv.reg) -> ()
  // SSR setup sequence for dimension 3
  "snitch.ssr_set_dimension_bound"(%bound) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<3>} : (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 160 : !riscv.reg
  // CHECK-NEXT: riscv_snitch.scfgw %bound, %{{.*}} {comment = "dm 0 dim 3 bound"} : (!riscv.reg, !riscv.reg) -> ()
  "snitch.ssr_set_dimension_stride"(%stride) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<3>} : (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 288 : !riscv.reg
  // CHECK-NEXT: riscv_snitch.scfgw %stride, %{{.*}} {comment = "dm 0 dim 3 stride"} : (!riscv.reg, !riscv.reg) -> ()
  "snitch.ssr_set_dimension_source"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<3>} : (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 864 : !riscv.reg
  // riscv_snitch.scfgw %addr, %{{.*}} {comment = "dm 0 dim 3 source"} : (!riscv.reg, !riscv.reg) -> ()
  "snitch.ssr_set_dimension_destination"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<3>} : (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 992 : !riscv.reg
  // CHECK-NEXT: riscv_snitch.scfgw %addr, %{{.*}} {comment = "dm 0 dim 3 destination"} : (!riscv.reg, !riscv.reg) -> ()
  "snitch.ssr_set_stream_repetition"(%rep) {"dm" = #builtin.int<0>}: (!riscv.reg) -> ()
  // CHECK: %{{.*}} = rv32.li 32 : !riscv.reg
  // CHECK-NEXT: riscv_snitch.scfgw %rep, %{{.*}} {comment = "dm 0 repeat"} : (!riscv.reg, !riscv.reg) -> ()
  // On/Off switching sequence
  "snitch.ssr_enable"() : () -> ()
  // CHECK: riscv.csrrsi 1984, 1
  "snitch.ssr_disable"() : () -> ()
  // CHECK: riscv.csrrci 1984, 1
}
