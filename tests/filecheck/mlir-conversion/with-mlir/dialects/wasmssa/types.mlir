// RUN: MLIR_ROUNDTRIP

"test.op"() {
  // CHECK: externref_val = !wasmssa.externref
  externref_val = !wasmssa.externref,

  // CHECK: funcref_val = !wasmssa.funcref
  funcref_val = !wasmssa.funcref
}: ()->()
