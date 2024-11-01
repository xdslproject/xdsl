// RUN: xdsl-opt %s | xdsl-opt | filecheck %s


"builtin.module"() ({
  "test.op"() {"value" = 42.0 : f32} : () -> ()
  // CHECK:      "test.op"() {"value" = 42.0 : f32} : () -> ()

  "test.op"() {"value" = -42.0 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {"value" = -42.0 : f32} : () -> ()

  "test.op"() {"value" = 34.e0 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {"value" = 34.0 : f32} : () -> ()

  "test.op"() {"value" = 34.e-23 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {"value" = 3.4e-22 : f32} : () -> ()

  "test.op"() {"value" = 34.e12 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {"value" = 34000000000000.0 : f32} : () -> ()

  "test.op"() {"value" = -34.e-12 : f32} : () -> ()
  // CHECK-NEXT: "test.op"() {"value" = -3.4e-11 : f32} : () -> ()
}) : () -> ()
