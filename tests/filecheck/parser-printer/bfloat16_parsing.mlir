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
}) : () -> ()
