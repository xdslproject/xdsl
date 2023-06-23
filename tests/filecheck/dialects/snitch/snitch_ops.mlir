// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s
"builtin.module"() ({
  %addr = "test.op"() : () -> !riscv.ireg<>
  %stream = "test.op"() : () -> !riscv.ireg<>
  %bound = "test.op"() : () -> !riscv.ireg<>
  %stride = "test.op"() : () -> !riscv.ireg<>
  %rep = "test.op"() : () -> !riscv.ireg<>
  // Usual SSR setup sequence:
  "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_stream_repetition"(%stream, %rep) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: "snitch.ssr_set_stream_repetition"(%stream, %rep) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_enable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_enable"() : () -> ()
  "snitch.ssr_disable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_disable"() : () -> ()
}) : () -> ()
