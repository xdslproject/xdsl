// RUN: XDSL_ROUNDTRIP

// CHECK: array_ssize = !wasmssa<local ref to i32>
"test.op"() {
  array_ssize = !wasmssa<local ref to i32>
}: ()->()

// CHECK: externref_val = !wasmssa.externref
"test.op"() {
  externref_val = !wasmssa.externref
}: ()->()

// CHECK: funcref_val = !wasmssa.funcref
"test.op"() {
  funcref_val = !wasmssa.funcref
}: ()->()

// CHECK: local_i64 = !wasmssa<local ref to i64>
"test.op"() {
  local_i64 = !wasmssa<local ref to i64>
}: ()->()

// CHECK: local_f32 = !wasmssa<local ref to f32>
"test.op"() {
  local_f32 = !wasmssa<local ref to f32>
}: ()->()

// CHECK: local_f64 = !wasmssa<local ref to f64>
"test.op"() {
  local_f64 = !wasmssa<local ref to f64>
}: ()->()
