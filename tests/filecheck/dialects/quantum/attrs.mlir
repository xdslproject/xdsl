// RUN: XDSL_ROUNDTRIP

"test.op"() {"angle" = !quantum.angle<0>} : () -> ()

// CHECK: "test.op"() {"angle" = !quantum.angle<0>} : () -> ()

"test.op"() {"angle" = !quantum.angle<pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = !quantum.angle<pi>} : () -> ()

"test.op"() {"angle" = !quantum.angle<2pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = !quantum.angle<0>} : () -> ()

"test.op"() {"angle" = !quantum.angle<pi:2>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = !quantum.angle<pi:2>} : () -> ()

"test.op"() {"angle" = !quantum.angle<3pi:2>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = !quantum.angle<3pi:2>} : () -> ()

"test.op"() {"angle" = !quantum.angle<5pi:2>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = !quantum.angle<pi:2>} : () -> ()

