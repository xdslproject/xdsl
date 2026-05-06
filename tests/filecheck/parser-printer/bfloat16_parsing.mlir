// RUN: XDSL_ROUNDTRIP


"builtin.module"() ({
  "test.op"() {"value" = 1.5 : bf16} : () -> ()
  // CHECK:      "test.op"() {value = 1.500000e+00 : bf16} : () -> ()

  "test.op"() {"value" = -2.0 : bf16} : () -> ()
  // CHECK-NEXT: "test.op"() {value = -2.000000e+00 : bf16} : () -> ()

  // 0.1 is not exactly representable in bf16; it must be quantised to
  // its nearest bf16 value on parse and print as the rounded form.
  "test.op"() {"value" = 0.1 : bf16} : () -> ()
  // CHECK-NEXT: "test.op"() {value = 1.000980e-01 : bf16} : () -> ()

  "test.op"() {"value" = 0x7fc0 : bf16} : () -> ()
  // CHECK-NEXT: "test.op"() {value = 0x7fc0 : bf16} : () -> ()

  "test.op"() {"value" = 0x7f80 : bf16} : () -> ()
  // CHECK-NEXT: "test.op"() {value = 0x7f80 : bf16} : () -> ()

  "test.op"() {"value" = 0xff80 : bf16} : () -> ()
  // CHECK-NEXT: "test.op"() {value = 0xff80 : bf16} : () -> ()

  "test.op"() {"weights" = array<bf16: 1.0, 2.0, -1.5>} : () -> ()
  // CHECK-NEXT: "test.op"() {weights = array<bf16: 1.000000e+00, 2.000000e+00, -1.500000e+00>} : () -> ()

  "test.op"() {"t" = dense<[1.0, 2.0]> : tensor<2xbf16>} : () -> ()
  // CHECK-NEXT: "test.op"() {t = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>} : () -> ()

  "test.op"() {"s" = dense<3.0> : tensor<4xbf16>} : () -> ()
  // CHECK-NEXT: "test.op"() {s = dense<3.000000e+00> : tensor<4xbf16>} : () -> ()
}) : () -> ()
