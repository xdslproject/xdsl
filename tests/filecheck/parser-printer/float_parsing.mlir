// RUN: xdsl-opt %s | xdsl-opt | filecheck %s


"builtin.module"() ({
  "test.op"() {"value" = 42.0 : f32} : () -> ()
  // CHECK:      "test.op"() {value = 4.200000e+01 : f32} : () -> ()

  "test.op"() {"value" = -42.0 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {value = -4.200000e+01 : f32} : () -> ()

  "test.op"() {"value" = 34.e0 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {value = 3.400000e+01 : f32} : () -> ()

  "test.op"() {"value" = 34.e-23 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {value = 3.400000e-22 : f32} : () -> ()

  "test.op"() {"value" = 34.e12 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {value = 3.400000e+13 : f32} : () -> ()

  "test.op"() {"value" = -34.e-12 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {value = -3.400000e-11 : f32} : () -> ()

  // this should print in full precision
  "test.op"() {"value" = 3.141592653589793 : f64} : () -> ()
  // CHECK-NEXT: "test.op"() {value = 3.1415926535897931 : f64} : () -> ()
}) : () -> ()
