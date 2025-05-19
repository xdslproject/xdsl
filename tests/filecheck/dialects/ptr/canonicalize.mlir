// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK-LABEL: @test_from_ptr_0
// CHECK-SAME: (%mr : memref<f32>)
func.func @test_from_ptr_0(%mr: memref<f32>) -> memref<f32> {
  // CHECK-NOT: ptr.to_ptr
  // CHECK-NOT: ptr.from_ptr
  // CHECK: return %mr
  %ptr = ptr_xdsl.to_ptr %mr : memref<f32> -> !ptr_xdsl.ptr
  %res = ptr_xdsl.from_ptr %ptr : !ptr_xdsl.ptr -> memref<f32>
  return %res : memref<f32>
}

// CHECK-LABEL: @test_from_ptr_1
// CHECK-SAME: (%mr : memref<f32>)
func.func @test_from_ptr_1(%mr: memref<f32>) -> !ptr_xdsl.ptr {
  // CHECK: return %ptr
  %ptr = ptr_xdsl.to_ptr %mr : memref<f32> -> !ptr_xdsl.ptr
  return %ptr : !ptr_xdsl.ptr
}

// CHECK-LABEL: @test_from_ptr_2
// CHECK-SAME: (%mr : memref<2x2xf32>)
func.func @test_from_ptr_2(%mr: memref<2x2xf32>) -> memref<4xf32> {
  // CHECK: return %res
  %ptr = ptr_xdsl.to_ptr %mr : memref<2x2xf32> -> !ptr_xdsl.ptr
  %res = ptr_xdsl.from_ptr %ptr : !ptr_xdsl.ptr -> memref<4xf32>
  return %res : memref<4xf32>
}

// CHECK-LABEL: @test_to_ptr_0
// CHECK-SAME: (%ptr : !ptr_xdsl.ptr)
func.func @test_to_ptr_0(%ptr: !ptr_xdsl.ptr) -> !ptr_xdsl.ptr {
  // CHECK-NOT: ptr.to_ptr
  // CHECK-NOT: ptr.from_ptr
  // CHECK: return %ptr
  %mr = ptr_xdsl.from_ptr %ptr : !ptr_xdsl.ptr -> memref<f32>
  %res = ptr_xdsl.to_ptr %mr : memref<f32> -> !ptr_xdsl.ptr
  return %res : !ptr_xdsl.ptr
}

// CHECK-LABEL: @test_to_ptr_1
// CHECK-SAME: (%ptr : !ptr_xdsl.ptr)
func.func @test_to_ptr_1(%ptr: !ptr_xdsl.ptr) -> memref<f32> {
  // CHECK: return %mr
  %mr = ptr_xdsl.from_ptr %ptr : !ptr_xdsl.ptr -> memref<f32>
  return %mr : memref<f32>
}
