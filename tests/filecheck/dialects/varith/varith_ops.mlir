// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%ia, %ib, %ic, %id = "test.op"() : () -> (i32, i32, i32, i32)
%fa, %fb, %fc, %fd = "test.op"() : () -> (f32, f32, f32, f32)
%ta, %tb, %tc, %td = "test.op"() : () -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)

%x1 = "varith.add"(%ia, %ib, %ic, %id) : (i32, i32, i32, i32) -> i32
// CHECK:  %x1 = varith.add %ia, %ib, %ic, %id : i32
// CHECK-GENERIC: %x1 = "varith.add"(%ia, %ib, %ic, %id) : (i32, i32, i32, i32) -> i32

%x2 = "varith.add"(%fa, %fb, %fc, %fd) : (f32, f32, f32, f32) -> f32
// CHECK:  %x2 = varith.add %fa, %fb, %fc, %fd : f32
// CHECK-GENERIC: %x2 = "varith.add"(%fa, %fb, %fc, %fd) : (f32, f32, f32, f32) -> f32

%x3 = "varith.add"(%ta, %tb, %tc, %td) : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK:  %x3 = varith.add %ta, %tb, %tc, %td : tensor<10xf32>
// CHECK-GENERIC: %x3 = "varith.add"(%ta, %tb, %tc, %td) : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>


%x4 = "varith.mul"(%ia, %ib, %ic, %id) : (i32, i32, i32, i32) -> i32
// CHECK:  %x4 = varith.mul %ia, %ib, %ic, %id : i32
// CHECK-GENERIC: %x4 = "varith.mul"(%ia, %ib, %ic, %id) : (i32, i32, i32, i32) -> i32

%x5 = "varith.mul"(%fa, %fb, %fc, %fd) : (f32, f32, f32, f32) -> f32
// CHECK:  %x5 = varith.mul %fa, %fb, %fc, %fd : f32
// CHECK-GENERIC: %x5 = "varith.mul"(%fa, %fb, %fc, %fd) : (f32, f32, f32, f32) -> f32

%x6 = "varith.mul"(%ta, %tb, %tc, %td) : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK:  %x6 = varith.mul %ta, %tb, %tc, %td : tensor<10xf32>
// CHECK-GENERIC: %x6 = "varith.mul"(%ta, %tb, %tc, %td) : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

%x7 = varith.switch %ia : i32 -> f32, [
  default: %fa,
  0: %fb,
  1: %fc,
  2: %fd
]
// CHECK:      %x7 = varith.switch %ia : i32 -> f32, [
// CHECK-NEXT:   default: %fa,
// CHECK-NEXT:   0: %fb,
// CHECK-NEXT:   1: %fc,
// CHECK-NEXT:   2: %fd
// CHECK-NEXT: ]

// CHECK-GENERIC: %x7 = "varith.switch"(%ia, %fa, %fb, %fc, %fd) <{case_values = dense<[0, 1, 2]> : vector<3xi32>}> : (i32, f32, f32, f32, f32) -> f32
