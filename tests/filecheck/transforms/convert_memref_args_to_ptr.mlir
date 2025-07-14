// RUN: xdsl-opt -p convert-memref-to-ptr{lower_func=true} --split-input-file --verify-diagnostics %s | filecheck %s

// CHECK:       builtin.module {

// CHECK-NEXT:    func.func @declaration(!ptr_xdsl.ptr) -> ()
func.func @declaration(%arg : memref<2x2xf32>)


// CHECK-NEXT:    func.func @simple(%arg : !ptr_xdsl.ptr) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
func.func @simple(%arg : memref<2x2xf32>) {
    func.return
}

// CHECK-NEXT:    func.func @id(%arg0 : !ptr_xdsl.ptr, %arg1 : !ptr_xdsl.ptr) -> (!ptr_xdsl.ptr, !ptr_xdsl.ptr) {
// CHECK-NEXT:      %arg0_1 = ptr_xdsl.from_ptr %arg0 : !ptr_xdsl.ptr -> memref<2x2xf32>
// CHECK-NEXT:      %arg1_1 = ptr_xdsl.from_ptr %arg1 : !ptr_xdsl.ptr -> memref<3x3xf32>
// CHECK-NEXT:      %arg0_2 = ptr_xdsl.to_ptr %arg0_1 : memref<2x2xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %arg1_2 = ptr_xdsl.to_ptr %arg1_1 : memref<3x3xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      func.return %arg0_2, %arg1_2 : !ptr_xdsl.ptr, !ptr_xdsl.ptr
// CHECK-NEXT:    }
func.func @id(%arg0 : memref<2x2xf32>, %arg1: memref<3x3xf32>) -> (memref<2x2xf32>, memref<3x3xf32>) {
    func.return %arg0, %arg1 : memref<2x2xf32>, memref<3x3xf32>
}

// CHECK-NEXT:    func.func @id2(%arg0 : !ptr_xdsl.ptr, %arg1 : !ptr_xdsl.ptr) -> (!ptr_xdsl.ptr, !ptr_xdsl.ptr) {
// CHECK-NEXT:      %arg0_1 = ptr_xdsl.from_ptr %arg0 : !ptr_xdsl.ptr -> memref<2x2xf32>
// CHECK-NEXT:      %arg1_1 = ptr_xdsl.from_ptr %arg1 : !ptr_xdsl.ptr -> memref<3x3xf32>
// CHECK-NEXT:      %arg0_2 = ptr_xdsl.to_ptr %arg0_1 : memref<2x2xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %arg1_2 = ptr_xdsl.to_ptr %arg1_1 : memref<3x3xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %resa, %resb = func.call @id(%arg0_2, %arg1_2) : (!ptr_xdsl.ptr, !ptr_xdsl.ptr) -> (!ptr_xdsl.ptr, !ptr_xdsl.ptr)
// CHECK-NEXT:      %resa_1 = ptr_xdsl.from_ptr %resa : !ptr_xdsl.ptr -> memref<2x2xf32>
// CHECK-NEXT:      %resb_1 = ptr_xdsl.from_ptr %resb : !ptr_xdsl.ptr -> memref<3x3xf32>
// CHECK-NEXT:      %resa_2 = ptr_xdsl.to_ptr %resa_1 : memref<2x2xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %resb_2 = ptr_xdsl.to_ptr %resb_1 : memref<3x3xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      func.return %resa_2, %resb_2 : !ptr_xdsl.ptr, !ptr_xdsl.ptr
// CHECK-NEXT:    }
func.func @id2(%arg0 : memref<2x2xf32>, %arg1 : memref<3x3xf32>) -> (memref<2x2xf32>, memref<3x3xf32>) {
    %resa, %resb = func.call @id(%arg0, %arg1) : (memref<2x2xf32>, memref<3x3xf32>) -> (memref<2x2xf32>, memref<3x3xf32>)
    func.return %resa, %resb : memref<2x2xf32>, memref<3x3xf32>
}

// CHECK-NEXT:    func.func @first(%arg : !ptr_xdsl.ptr) -> f32 {
// CHECK-NEXT:      %arg_1 = ptr_xdsl.from_ptr %arg : !ptr_xdsl.ptr -> memref<2x2xf32>
// CHECK-NEXT:      %pointer = ptr_xdsl.to_ptr %arg_1 : memref<2x2xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %res = ptr_xdsl.load %pointer : !ptr_xdsl.ptr -> f32
// CHECK-NEXT:      func.return %res : f32
// CHECK-NEXT:    }
func.func @first(%arg : memref<2x2xf32>) -> f32 {
    %pointer = ptr_xdsl.to_ptr %arg : memref<2x2xf32> -> !ptr_xdsl.ptr
    %res = ptr_xdsl.load %pointer : !ptr_xdsl.ptr -> f32
    func.return %res : f32
}

// CHECK-NEXT:  }
