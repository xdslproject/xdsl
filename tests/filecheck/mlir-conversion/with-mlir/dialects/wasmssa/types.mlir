// RUN: MLIR_ROUNDTRIP

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

// CHECK: limit_min = !wasmssa<limit[2:]>
"test.op"() {
  limit_min = !wasmssa<limit[2:]>
}: ()->()

// CHECK: limit_min_max = !wasmssa<limit[0: 65536]>
"test.op"() {
  limit_min_max = !wasmssa<limit[0: 65536]>
}: ()->()

// CHECK: table_funcref = !wasmssa<tabletype !wasmssa.funcref [348:]>
"test.op"() {
  table_funcref = !wasmssa<tabletype !wasmssa.funcref [348:]>
}: ()->()

// CHECK: table_externref = !wasmssa<tabletype !wasmssa.externref [0: 65536]>
"test.op"() {
  table_externref = !wasmssa<tabletype !wasmssa.externref [0: 65536]>
}: ()->()
