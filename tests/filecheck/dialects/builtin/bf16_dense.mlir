// RUN: XDSL_ROUNDTRIP


"builtin.module"() ({
  "test.op"() {"weights" = array<bf16: 1.0, 2.0, -1.5>} : () -> ()
  // CHECK:      "test.op"() {weights = array<bf16: 1.000000e+00, 2.000000e+00, -1.500000e+00>} : () -> ()

  "test.op"() {"t" = dense<[1.0, 2.0]> : tensor<2xbf16>} : () -> ()
  // CHECK-NEXT: "test.op"() {t = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>} : () -> ()

  "test.op"() {"s" = dense<3.0> : tensor<4xbf16>} : () -> ()
  // CHECK-NEXT: "test.op"() {s = dense<3.000000e+00> : tensor<4xbf16>} : () -> ()
}) : () -> ()
