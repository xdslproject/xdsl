// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %addr = "test.op"() : () -> !riscv.reg<>
  %stream = "test.op"() : () -> !riscv.reg<>
  %bound = "test.op"() : () -> !riscv.reg<>
  %stride = "test.op"() : () -> !riscv.reg<>
  %rep = "test.op"() : () -> !riscv.reg<>
  // Usual SSR setup sequence:
  "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_set_stream_repetition"(%stream, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: "snitch.ssr_set_stream_repetition"(%stream, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_enable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_enable"() : () -> ()
  "snitch.ssr_disable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_disable"() : () -> ()
}) : () -> ()
