// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %A, %B = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>) // vector {A,B}: base address
  %n = "test.op"() : () -> !riscv.reg<> // vector {A,B}: size
  // dm: data mover id
  %s0 = "test.op"() : () -> !riscv.reg<>
  %s1 = "test.op"() : () -> !riscv.reg<>
  // bound: vector size, minus one
  %bound = "test.op"() : () -> !riscv.reg<>
  // stride: in bytes
  %stride = "test.op"() : () -> !riscv.reg<>
  // repetition: number of times each element will be repeated, minus one
  %rep = "test.op"() : () -> !riscv.reg<>
  // Usual SSR setup sequence:
  "snitch.ssr_setup_shape"(%s0, %bound, %stride) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: "snitch.ssr_setup_shape"(%s0, %bound, %stride) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_setup_repetition"(%s0, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_setup_repetition"(%s0, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_read"(%s0, %A) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_read"(%s0, %A) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_setup_shape"(%s1, %bound, %stride) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_setup_shape"(%s1, %bound, %stride) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_setup_repetition"(%s1, %rep) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_setup_repetition"(%s1, %rep) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_read"(%s1, %B) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_read"(%s1, %B) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_enable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_enable"() : () -> ()
  "snitch.ssr_disable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_disable"() : () -> ()
}) : () -> ()
