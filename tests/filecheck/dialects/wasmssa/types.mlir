// RUN: XDSL_ROUNDTRIP

"test.op"() {
  // CHECK: externref_val = !wasmssa.externref
  externref_val = !wasmssa.externref,

  // CHECK: funcref_val = !wasmssa.funcref
  funcref_val = !wasmssa.funcref,

  // CHECK: array_ssize = !wasmssa<local ref to i32>
  array_ssize = !wasmssa<local ref to i32>,

  // CHECK: local_i64 = !wasmssa<local ref to i64>
  local_i64 = !wasmssa<local ref to i64>,

  // CHECK: local_f32 = !wasmssa<local ref to f32>
  local_f32 = !wasmssa<local ref to f32>,

  // CHECK: local_f64 = !wasmssa<local ref to f64>
  local_f64 = !wasmssa<local ref to f64>,

  // CHECK: limit_min = !wasmssa<limit[2:]>
  limit_min = !wasmssa<limit[2:]>,

  // CHECK: limit_min_max = !wasmssa<limit[0: 65536]>
  limit_min_max = !wasmssa<limit[0: 65536]>
}: ()->()
