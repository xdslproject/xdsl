// RUN: XDSL_ROUNDTRIP

// CHECK: entry1 = #dlti.dl_entry<"str", i32>
// CHECK: entry2 = #dlti.dl_entry<i32, i32>
"test.op"() {
    entry1 = #dlti.dl_entry<"str", i32>,
    entry2 = #dlti.dl_entry<i32, i32>
} : () -> ()

// -----

// CHECK: spec1 = #dlti.dl_spec<>
// CHECK: spec2 = #dlti.dl_spec<"str" = i32, i32 = i32>
"test.op"() {
    spec1 = #dlti.dl_spec<>,
    spec2 = #dlti.dl_spec<"str" = i32, i32 = i32>
} : () -> ()

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" = #dlti.target_device_spec<
// CHECK-SAME:      "L1_cache_size_in_bytes" = 4096 : i32>,
// CHECK-SAME:    "GPU" = #dlti.target_device_spec<
// CHECK-SAME:      "max_vector_op_width" = 128 : i32>
// CHECK-SAME:  >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "L1_cache_size_in_bytes" = 4096 : i32>,
    "GPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 128 : i32>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" = #dlti.target_device_spec<
// CHECK-SAME:      "L1_cache_size_in_bytes" = 4096 : i32>,
// CHECK-SAME:    "GPU" = #dlti.target_device_spec<
// CHECK-SAME:      "L1_cache_size_in_bytes" = 8192 : i32>
// CHECK-SAME:  >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "L1_cache_size_in_bytes" = 4096 : i32>,
    "GPU" = #dlti.target_device_spec<
      "L1_cache_size_in_bytes" = 8192 : i32>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = 4096 : i64>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = 8192 : i64>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "L1_cache_size_in_bytes" = 4096 : i64>,
    "GPU" = #dlti.target_device_spec<
      "L1_cache_size_in_bytes" = 8192 : i64>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 64 : i32>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 128 : i32>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 64 : i32>,
    "GPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 128 : i32>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 64 : i64>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 128 : i64>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 64 : i64>,
    "GPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 128 : i64>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 64 : i64>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 128 : i64>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 64 : i64>,
    "GPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 128 : i64>
  >} {}

// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = 4096 : ui32>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = "128">
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "L1_cache_size_in_bytes" = 4096 : ui32>,
    "GPU" = #dlti.target_device_spec<
      "max_vector_op_width" = "128">
  >} {}

// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 4.096000e+03 : f32>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = "128">
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 4096.0 : f32>,
    "GPU" = #dlti.target_device_spec<
      "L1_cache_size_in_bytes" = "128">
  >} {}


// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = 4096 : ui32>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 128 : i64>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<"L1_cache_size_in_bytes" = 4096 : ui32>,
    "GPU" = #dlti.target_device_spec<"max_vector_op_width" = 128>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.map = #dlti.map<
// CHECK-SAME:     "magic_num" = 42 : i32,
// CHECK-SAME:     "magic_num_float" = 4.242000e+01 : f32,
// CHECK-SAME:     "magic_type" = i32,
// CHECK-SAME:     i32 = #dlti.map<"bitwidth" = 32 : i32>
// CHECK:        >} {
// CHECK:      }
module attributes {
  dlti.map = #dlti.map<"magic_num" = 42 : i32,
                       "magic_num_float" = 42.42 : f32,
                       "magic_type" = i32,
                        i32 = #dlti.map<"bitwidth" = 32 : i32>>
  } {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.map = #dlti.map<
// CHECK-SAME:     "CPU" = #dlti.map<"L1_cache_size_in_bytes" = 4096 : i32>,
// CHECK-SAME:     "GPU" = #dlti.map<"max_vector_op_width" = 128 : i32>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.map = #dlti.map<
    "CPU" = #dlti.map<"L1_cache_size_in_bytes" = 4096 : i32>,
    "GPU" = #dlti.map<"max_vector_op_width" = 128 : i32>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "key" = #dlti.map<"V1" = 22 : i32, "V2" = 100 : i32,
// CHECK-SAME:                          "V3" = 22 : i32>>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "key" = #dlti.map<"V1" = 24 : i32, "V2" = 16 : i32,
// CHECK-SAME:                          "V3" = 9.920000e+01 : f32>>
// CHECK-SAME:   >} {
// CHECK:      }

module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<"key" = #dlti.map<"V1" = 22 : i32, "V2" = 100 : i32, "V3" = 22 : i32>>,
    "GPU" = #dlti.target_device_spec<"key" = #dlti.map<"V1" = 24 : i32, "V2" = 16 : i32, "V3" = 99.2 : f32>>
  >} {}
