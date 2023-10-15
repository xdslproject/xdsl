// RUN: xdsl-opt -p lower-snitch %s | filecheck %s
builtin.module {
  %addr = "test.op"() : () -> !riscv.reg<>
  %stream = riscv.li 0 : () -> !riscv.reg<>
  %bound = riscv.li 9 : () -> !riscv.reg<>
  %stride = riscv.li 4 : () -> !riscv.reg<>
  %rep = riscv.li 0 : () -> !riscv.reg<>
  // SSR setup sequence for dimension 0
  "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 64 : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv.scfgw %bound, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 192 : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} riscv.scfgw %stride, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 768 : (!riscv.reg<>) -> !riscv.reg<>
  // %{{.*}} = riscv.scfgw %addr, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 896 : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv.scfgw %addr, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  // SSR setup sequence for dimension 3
  "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 3 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 160 : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv.scfgw %bound, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 3 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 288 : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv.scfgw %stride, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 3 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 864 : (!riscv.reg<>) -> !riscv.reg<>
  // riscv.scfgw %addr, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 3 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 992 : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv.scfgw %addr, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  "snitch.ssr_set_stream_repetition"(%stream, %rep) : (!riscv.reg<>, !riscv.reg<>) -> ()
  // CHECK: %{{.*}} = riscv.addi %stream, 32 : (!riscv.reg<>) -> !riscv.reg<>
  // CHECK-NEXT: %{{.*}} = riscv.scfgw %rep, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<zero>
  // On/Off switching sequence
  "snitch.ssr_enable"() : () -> ()
  // CHECK: riscv.csrrsi 1984, 1
  "snitch.ssr_disable"() : () -> ()
  // CHECK: riscv.csrrci 1984, 1
}
