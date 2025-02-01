// RUN: xdsl-opt -p convert-memref-to-ptr{convert_func_args=true} --split-input-file --verify-diagnostics %s | filecheck %s

func.func @declaration(%arg : memref<2x2xf32>)

func.func @simple(%arg : memref<2x2xf32>) {
    func.return
}

func.func @id(%arg : memref<2x2xf32>) -> memref<2x2xf32> {
    func.return %arg : memref<2x2xf32>
}

func.func @id2(%arg : memref<2x2xf32>) -> memref<2x2xf32> {
    %res = func.call @id(%arg) : (memref<2x2xf32>) -> memref<2x2xf32>
    func.return %res : memref<2x2xf32>
}

func.func @first(%arg : memref<2x2xf32>) -> f32 {
    %pointer = ptr_xdsl.to_ptr %arg : memref<2x2xf32> -> !ptr_xdsl.ptr
    %res = ptr_xdsl.load %pointer : !ptr_xdsl.ptr -> f32
    func.return %res : f32
}
