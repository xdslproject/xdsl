// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
"builtin.module"() ({
  %A = "test.op"() : () -> !riscv.reg<> // vector A: base address
  %B = "test.op"() : () -> !riscv.reg<> // vector B: base address
  %n = "riscv.li"() {"immediate" = 10 : i32} : () -> !riscv.reg<> // vector {A,B}: size
  // dm: data mover id
  %s0 = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<>
  %s1 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
  // bound: vector size, minus one
  %bound = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.reg<>
  // stride: in bytes
  %stride = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.reg<>
  // repetition: number of times each element will be repeated, minus one
  %rep = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<>
  // dimension: dimension index, minus one
  %dim = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<>
  
  "snitch.ssr_setup_bound_stride_1d"(%s0, %bound, %stride) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: "snitch.ssr_setup_bound_stride_1d"(%s0, %bound, %stride) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_setup_repetition"(%s0, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_setup_repetition"(%s0, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_read"(%s0, %dim, %A) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_read"(%s0, %dim, %A) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_setup_bound_stride_1d"(%s1, %bound, %stride) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_setup_bound_stride_1d"(%s1, %bound, %stride) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_setup_repetition"(%s1, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_setup_repetition"(%s1, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_read"(%s1, %dim, %B) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK-NEXT: "snitch.ssr_read"(%s1, %dim, %B) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
  "snitch.ssr_enable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_enable"() : () -> ()
  "snitch.ssr_disable"() : () -> ()
  // CHECK-NEXT: "snitch.ssr_disable"() : () -> ()
}) : () -> ()
