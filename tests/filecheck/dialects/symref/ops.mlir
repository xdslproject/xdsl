// RUN: XDSL_ROUNDTRIP

// CHECK:          symref.declare "counter"
// CHECK-NEXT:     symref.declare "tensor_ref"
symref.declare "counter"
symref.declare "tensor_ref"

// CHECK-NEXT:     %{{.*}} = symref.fetch @counter : i32
// CHECK-NEXT:     %{{.*}} = symref.fetch @tensor_ref : tensor<4x4xf32>
%0 = symref.fetch @counter : i32
%1 = symref.fetch @tensor_ref : tensor<4x4xf32>

// CHECK-NEXT:     %{{.*}} = "test.op"() : () ->  i32
// CHECK-NEXT:     %{{.*}} = "test.op"() : () -> tensor<4x4xf32>
%constant = "test.op"() : () -> i32
%tensor = "test.op"() : () -> tensor<4x4xf32>

// CHECK-NEXT:     symref.update @counter = %{{.*}} : i32
// CHECK-NEXT:     symref.update @tensor_ref = %{{.*}} : tensor<4x4xf32>
symref.update @counter = %constant : i32
symref.update @tensor_ref = %tensor : tensor<4x4xf32>
