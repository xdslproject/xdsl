builtin.module {
  func.func @fixed_matmul(%0 : memref<16x16xf32>, %1 : memref<16x16xf32>, %2 : memref<16x16xf32>) {
    %3 = arith.constant 0 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.constant 16 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.constant 1 : index
    %8 = arith.constant 0.000000e+00 : f32
    scf.for %9 = %4 to %6 step %7 {
      scf.for %10 = %4 to %6 step %7 {
        memref.store %8, %0[%9, %10] : memref<16x16xf32>
        scf.for %11 = %4 to %6 step %7 {
          %12 = memref.load %1[%9, %11] : memref<16x16xf32>
          %13 = memref.load %2[%11, %10] : memref<16x16xf32>
          %14 = arith.mulf %12, %13 : f32
          %15 = memref.load %0[%9, %10] : memref<16x16xf32>
          %16 = arith.addf %15, %14 : f32
          memref.store %16, %0[%9, %10] : memref<16x16xf32>
        }
      }
    }
    func.return
  }
}

builtin.module {
  func.func @fixed_matmul(%0 : !ptr_xdsl.ptr, %1 : !ptr_xdsl.ptr, %2 : !ptr_xdsl.ptr) {
    %3 = arith.constant 0 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.constant 16 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.constant 1 : index
    %8 = arith.constant 0.000000e+00 : f32
    scf.for %9 = %4 to %6 step %7 {
      scf.for %10 = %4 to %6 step %7 {
        %pointer_dim_stride = arith.constant 16 : index
        %pointer_dim_offset = arith.muli %9, %pointer_dim_stride : index
        %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %10 : index
        %bytes_per_element = ptr_xdsl.type_offset f32 : index
        %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
        %offset_pointer = ptr_xdsl.ptradd %0, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
        ptr_xdsl.store %8, %offset_pointer : f32, !ptr_xdsl.ptr
        scf.for %11 = %4 to %6 step %7 {
          %pointer_dim_stride_2 = arith.constant 16 : index
          %pointer_dim_offset_1 = arith.muli %9, %pointer_dim_stride_2 : index
          %pointer_dim_stride_3 = arith.addi %pointer_dim_offset_1, %11 : index
          %bytes_per_element_1 = ptr_xdsl.type_offset f32 : index
          %scaled_pointer_offset_1 = arith.muli %pointer_dim_stride_3, %bytes_per_element_1 : index
          %offset_pointer_1 = ptr_xdsl.ptradd %1, %scaled_pointer_offset_1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
          %12 = ptr_xdsl.load %offset_pointer_1 : !ptr_xdsl.ptr -> f32
          %pointer_dim_stride_4 = arith.constant 16 : index
          %pointer_dim_offset_2 = arith.muli %11, %pointer_dim_stride_4 : index
          %pointer_dim_stride_5 = arith.addi %pointer_dim_offset_2, %10 : index
          %bytes_per_element_2 = ptr_xdsl.type_offset f32 : index
          %scaled_pointer_offset_2 = arith.muli %pointer_dim_stride_5, %bytes_per_element_2 : index
          %offset_pointer_2 = ptr_xdsl.ptradd %2, %scaled_pointer_offset_2 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
          %13 = ptr_xdsl.load %offset_pointer_2 : !ptr_xdsl.ptr -> f32
          %14 = arith.mulf %12, %13 : f32
          %pointer_dim_stride_6 = arith.constant 16 : index
          %pointer_dim_offset_3 = arith.muli %9, %pointer_dim_stride_6 : index
          %pointer_dim_stride_7 = arith.addi %pointer_dim_offset_3, %10 : index
          %bytes_per_element_3 = ptr_xdsl.type_offset f32 : index
          %scaled_pointer_offset_3 = arith.muli %pointer_dim_stride_7, %bytes_per_element_3 : index
          %offset_pointer_3 = ptr_xdsl.ptradd %0, %scaled_pointer_offset_3 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
          %15 = ptr_xdsl.load %offset_pointer_3 : !ptr_xdsl.ptr -> f32
          %16 = arith.addf %15, %14 : f32
          %pointer_dim_stride_8 = arith.constant 16 : index
          %pointer_dim_offset_4 = arith.muli %9, %pointer_dim_stride_8 : index
          %pointer_dim_stride_9 = arith.addi %pointer_dim_offset_4, %10 : index
          %bytes_per_element_4 = ptr_xdsl.type_offset f32 : index
          %scaled_pointer_offset_4 = arith.muli %pointer_dim_stride_9, %bytes_per_element_4 : index
          %offset_pointer_4 = ptr_xdsl.ptradd %0, %scaled_pointer_offset_4 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
          ptr_xdsl.store %16, %offset_pointer_4 : f32, !ptr_xdsl.ptr
        }
      }
    }
    func.return
  }
}
