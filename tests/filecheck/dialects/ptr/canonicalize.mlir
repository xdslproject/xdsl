// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK-LABEL: @test_from_ptr_0
// CHECK-SAME: (%mr : memref<f32>)
func.func @test_from_ptr_0(%mr: memref<f32>) -> memref<f32> {
  // CHECK-NOT: ptr.to_ptr
  // CHECK-NOT: ptr.from_ptr
  // CHECK: return %mr
  %ptr = ptr_xdsl.to_ptr %mr : memref<f32> -> !ptr_xdsl.ptr
  %res = ptr_xdsl.from_ptr %ptr : !ptr_xdsl.ptr -> memref<f32>
  return %mr : memref<f32>
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
