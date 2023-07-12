// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s
"builtin.module"() ({
  %addr = "test.op"() : () -> !riscv.reg<x$>
  %stream = "test.op"() : () -> !riscv.reg<x$>
  %bound = "test.op"() : () -> !riscv.reg<x$>
  %stride = "test.op"() : () -> !riscv.reg<x$>
  %rep = "test.op"() : () -> !riscv.reg<x$>
  // Usual SSR setup sequence:
  "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 0 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK: "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 0 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 0 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK: "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 0 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK: "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK: "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "snitch.ssr_set_stream_repetition"(%stream, %rep) : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  // CHECK: "snitch.ssr_set_stream_repetition"(%stream, %rep) : (!riscv.reg<x$>, !riscv.reg<x$>) -> ()
  "snitch.ssr_enable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_enable"() : () -> ()
  "snitch.ssr_disable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_disable"() : () -> ()
}) : () -> ()
