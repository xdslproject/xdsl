// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | filecheck %s

// CHECK: entry1 = #dlti.dl_entry<"str", i32>
// CHECK: entry2 = #dlti.dl_entry<i32, i32>
"test.op"() {
    entry1 = #dlti.dl_entry<"str", i32>,
    entry2 = #dlti.dl_entry<i32, i32>
} : () -> ()

// CHECK: spec1 = #dlti.dl_spec<>
// CHECK: spec2 = #dlti.dl_spec<"str" = i32, i32 = i32>
"test.op"() {
    spec1 = #dlti.dl_spec<>,
    spec2 = #dlti.dl_spec<"str" = i32, i32 = i32>
} : () -> ()

// CHECK: "builtin.module"() ({
// CHECK:   ^bb0:
// CHECK:   }) {
// CHECK:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK:          "CPU" = #dlti.target_device_spec<
// CHECK:                   "dlti.L1_cache_size_in_bytes" = 4096 : ui32>,
// CHECK:          "GPU" = #dlti.target_device_spec<
// CHECK:                   "dlti.max_vector_op_width" = 128 : ui32>>} : () -> ()
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "dlti.L1_cache_size_in_bytes" = 4096 : ui32>,
    "GPU" = #dlti.target_device_spec<
      "dlti.max_vector_op_width" = 128 : ui32>
  >} {}

// CHECK: "builtin.module"() ({
// CHECK:   ^bb0:
// CHECK:   }) {
// CHECK:   dlti.map = #dlti.map<
// CHECK:          "CPU" = #dlti.map<
// CHECK:                   "L1_cache_size_in_bytes" = 4096 : i32>,
// CHECK:          "GPU" = #dlti.map<
// CHECK:                   "max_vector_op_width" = 128 : i32>>} : () -> ()
module attributes {
  dlti.map = #dlti.map<
    "CPU" = #dlti.map<"L1_cache_size_in_bytes" = 4096 : i32>,
    "GPU" = #dlti.map<"max_vector_op_width" = 128 : i32>
  >} {}
