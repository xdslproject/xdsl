// RUN: MLIR_GENERIC_ROUNDTRIP

"builtin.module"() ({
  "test.op"() {"value" = 1.5 : bf16} : () -> ()
  // CHECK:      "test.op"() {value = 1.500000e+00 : bf16} : () -> ()

  "test.op"() {"value" = -2.0 : bf16} : () -> ()
  // CHECK-NEXT: "test.op"() {value = -2.000000e+00 : bf16} : () -> ()

  "test.op"() {"weights" = dense<[1.0, 2.0, -1.5]> : tensor<3xbf16>} : () -> ()
  // CHECK-NEXT: "test.op"() {weights = dense<[1.000000e+00, 2.000000e+00, -1.500000e+00]> : tensor<3xbf16>} : () -> ()

  "test.op"() {"splat" = dense<3.0> : tensor<4xbf16>} : () -> ()
  // CHECK-NEXT: "test.op"() {splat = dense<3.000000e+00> : tensor<4xbf16>} : () -> ()
}) : () -> ()
