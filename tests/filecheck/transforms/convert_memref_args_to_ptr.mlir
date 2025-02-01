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

// CHECK-NEXT:    func.func @id(%arg : !ptr_xdsl.ptr) -> !ptr_xdsl.ptr {
// CHECK-NEXT:      func.return %arg : !ptr_xdsl.ptr
// CHECK-NEXT:    }
func.func @id(%arg : memref<2x2xf32>) -> memref<2x2xf32> {
    func.return %arg : memref<2x2xf32>
}

// CHECK-NEXT:    func.func @id2(%arg : !ptr_xdsl.ptr) -> !ptr_xdsl.ptr {
// CHECK-NEXT:      %res = func.call @id(%arg) : (!ptr_xdsl.ptr) -> !ptr_xdsl.ptr
// CHECK-NEXT:      func.return %res : !ptr_xdsl.ptr
// CHECK-NEXT:    }
func.func @id2(%arg : memref<2x2xf32>) -> memref<2x2xf32> {
    %res = func.call @id(%arg) : (memref<2x2xf32>) -> memref<2x2xf32>
    func.return %res : memref<2x2xf32>
}

// CHECK-NEXT:    func.func @first(%arg : !ptr_xdsl.ptr) -> f32 {
// CHECK-NEXT:      %res = ptr_xdsl.load %arg : !ptr_xdsl.ptr -> f32
// CHECK-NEXT:      func.return %res : f32
// CHECK-NEXT:    }
func.func @first(%arg : memref<2x2xf32>) -> f32 {
    %pointer = ptr_xdsl.to_ptr %arg : memref<2x2xf32> -> !ptr_xdsl.ptr
    %res = ptr_xdsl.load %pointer : !ptr_xdsl.ptr -> f32
    func.return %res : f32
}

// CHECK-NEXT:  }
