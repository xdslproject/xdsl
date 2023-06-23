// RUN: xdsl-opt -p lower-snitch %s --print-op-generic | filecheck %s
"builtin.module"() ({
  %addr = "test.op"() : () -> !riscv.ireg<>
  %stream = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.ireg<>
  %bound = "riscv.li"() {"immediate" = 9 : i32} : () -> !riscv.ireg<>
  %stride = "riscv.li"() {"immediate" = 4 : i32} : () -> !riscv.ireg<>
  %rep = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.ireg<>
  // SSR setup sequence for dimension 0
  "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 64 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: "riscv.scfgw"(%bound, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 192 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: "riscv.scfgw"(%stride, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 768 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // "riscv.scfgw"(%addr, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 0 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 896 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: "riscv.scfgw"(%addr, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // SSR setup sequence for dimension 3
  "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 3 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 160 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: "riscv.scfgw"(%bound, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_stride"(%stream, %stride) {"dimension" = 3 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 288 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: "riscv.scfgw"(%stride, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_source"(%stream, %addr) {"dimension" = 3 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 864 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // "riscv.scfgw"(%addr, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_dimension_destination"(%stream, %addr) {"dimension" = 3 : i32} : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 992 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: "riscv.scfgw"(%addr, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  "snitch.ssr_set_stream_repetition"(%stream, %rep) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // CHECK: %{{.*}} = "riscv.addi"(%stream) {"immediate" = 32 : i32} : (!riscv.ireg<>) -> !riscv.ireg<>
  // CHECK-NEXT: "riscv.scfgw"(%rep, %{{.*}}) : (!riscv.ireg<>, !riscv.ireg<>) -> ()
  // On/Off switching sequence
  "snitch.ssr_enable"() : () -> ()
  // CHECK: "riscv.csrrsi"() {"csr" = 1984 : i32, "immediate" = 1 : i32}
  "snitch.ssr_disable"() : () -> ()
  // CHECK: "riscv.csrrci"() {"csr" = 1984 : i32, "immediate" = 1 : i32}
}) : () -> ()
