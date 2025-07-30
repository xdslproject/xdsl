// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s --strict-whitespace

// CHECK: key must be a string or a type attribute
"test.op"() {
    a = #dlti.dl_entry<9 : i32, i32>
} : () -> ()

// -----

// CHECK: duplicate DLTI entry key

"test.op"() {
  spec2 = #dlti.dl_spec<i32 = i32, i32 = i32>
>} : () -> ()

// -----

// CHECK: key must be a string or a type attribute
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    0 = #dlti.target_device_spec<
        "L1_cache_size_in_bytes" = 4096 : i32>
  >} {}

// -----

// CHECK: duplicate DLTI entry key
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
            "L1_cache_size_in_bytes" = 4096>,
    "CPU" = #dlti.target_device_spec<
            "L1_cache_size_in_bytes" = 8192>
  >} {}

// -----

// CHECK: duplicate DLTI entry key

module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<"L1_cache_size_in_bytes" = 4096,
                                     "L1_cache_size_in_bytes" = 8192>
  >} {}
