// RUN: XDSL_ROUNDTRIP
"builtin.module"() ({
  %addr = "test.op"() : () -> !riscv.reg
  %bound = "test.op"() : () -> !riscv.reg
  %stride = "test.op"() : () -> !riscv.reg
  %rep = "test.op"() : () -> !riscv.reg
  // Usual SSR setup sequence:
  "snitch.ssr_set_dimension_bound"(%bound) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: "snitch.ssr_set_dimension_bound"(%bound) {dm = #builtin.int<0>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
  "snitch.ssr_set_dimension_stride"(%stride) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: "snitch.ssr_set_dimension_stride"(%stride) {dm = #builtin.int<0>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
  "snitch.ssr_set_dimension_source"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: "snitch.ssr_set_dimension_source"(%addr) {dm = #builtin.int<0>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
  "snitch.ssr_set_dimension_destination"(%addr) {"dm" = #builtin.int<0>, "dimension" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: "snitch.ssr_set_dimension_destination"(%addr) {dm = #builtin.int<0>, dimension = #builtin.int<0>} : (!riscv.reg) -> ()
  "snitch.ssr_set_stream_repetition"(%rep) {"dm" = #builtin.int<0>} : (!riscv.reg) -> ()
  // CHECK: "snitch.ssr_set_stream_repetition"(%rep) {dm = #builtin.int<0>} : (!riscv.reg) -> ()
  "snitch.ssr_enable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_enable"() : () -> ()
  "snitch.ssr_disable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_disable"() : () -> ()
}) : () -> ()
