// RUN: xdsl-opt -p convert-vector-to-ptr --split-input-file %s | filecheck %s

%m = "test.op"(): () -> memref<16xf32>
%i = arith.constant 0: index
%v = vector.load %m[%i]: memref<16xf32>, vector<8xf32>

// CHECK:       builtin.module {
// CHECK-NEXT:    %m = "test.op"() : () -> memref<16xf32>
// CHECK-NEXT:    %i = arith.constant 0 : index
// CHECK-NEXT:    %m_1 = ptr_xdsl.to_ptr %m : memref<16xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:    %bytes_per_element = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:    %scaled_pointer_offset = arith.muli %i, %bytes_per_element : index
// CHECK-NEXT:    %offset_pointer = ptr_xdsl.ptradd %m_1, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:    %v = ptr_xdsl.load %offset_pointer : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK-NEXT:  }

// -----

%m0 = "test.op"(): () -> memref<2x8xf32>
%i0 = arith.constant 0: index
%j0 = arith.constant 0: index
%v0 = vector.load %m0[%i0,%j0]: memref<2x8xf32>, vector<8xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m0 = "test.op"() : () -> memref<2x8xf32>
// CHECK-NEXT:   %i0 = arith.constant 0 : index
// CHECK-NEXT:   %j0 = arith.constant 0 : index
// CHECK-NEXT:   %m0_1 = ptr_xdsl.to_ptr %m0 : memref<2x8xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %pointer_dim_stride = arith.constant 8 : index
// CHECK-NEXT:   %pointer_dim_offset = arith.muli %i0, %pointer_dim_stride : index
// CHECK-NEXT:   %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %j0 : index
// CHECK-NEXT:   %bytes_per_element = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:   %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
// CHECK-NEXT:   %offset_pointer = ptr_xdsl.ptradd %m0_1, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v0 = ptr_xdsl.load %offset_pointer : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK-NEXT: }

// -----

%m1 = "test.op"(): () -> memref<2x8xf16>
%i1 = arith.constant 0: index
%j1 = arith.constant 0: index
%v1 = vector.load %m1[%i1,%j1]: memref<2x8xf16>, vector<8xf16>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m1 = "test.op"() : () -> memref<2x8xf16>
// CHECK-NEXT:   %i1 = arith.constant 0 : index
// CHECK-NEXT:   %j1 = arith.constant 0 : index
// CHECK-NEXT:   %m1_1 = ptr_xdsl.to_ptr %m1 : memref<2x8xf16> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %pointer_dim_stride = arith.constant 8 : index
// CHECK-NEXT:   %pointer_dim_offset = arith.muli %i1, %pointer_dim_stride : index
// CHECK-NEXT:   %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %j1 : index
// CHECK-NEXT:   %bytes_per_element = ptr_xdsl.type_offset f16 : index
// CHECK-NEXT:   %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
// CHECK-NEXT:   %offset_pointer = ptr_xdsl.ptradd %m1_1, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v1 = ptr_xdsl.load %offset_pointer : !ptr_xdsl.ptr -> vector<8xf16>
// CHECK-NEXT: }

// -----

%m2 = "test.op"(): () -> memref<2x32xf32>
%i2 = arith.constant 0: index
%j2 = arith.constant 0: index
%v2 = vector.load %m2[%i2,%j2]: memref<2x32xf32>, vector<8xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m2 = "test.op"() : () -> memref<2x32xf32>
// CHECK-NEXT:   %i2 = arith.constant 0 : index
// CHECK-NEXT:   %j2 = arith.constant 0 : index
// CHECK-NEXT:   %m2_1 = ptr_xdsl.to_ptr %m2 : memref<2x32xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %pointer_dim_stride = arith.constant 32 : index
// CHECK-NEXT:   %pointer_dim_offset = arith.muli %i2, %pointer_dim_stride : index
// CHECK-NEXT:   %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %j2 : index
// CHECK-NEXT:   %bytes_per_element = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:   %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
// CHECK-NEXT:   %offset_pointer = ptr_xdsl.ptradd %m2_1, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v2 = ptr_xdsl.load %offset_pointer : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK-NEXT: }

// -----

%m0 = "test.op"(): () -> memref<2x8xf32>
%i0 = arith.constant 0: index
%j0 = arith.constant 0: index
%v0 = "test.op"(): () -> vector<8xf32>
vector.store %v0, %m0[%i0,%j0]: memref<2x8xf32>, vector<8xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m0 = "test.op"() : () -> memref<2x8xf32>
// CHECK-NEXT:   %i0 = arith.constant 0 : index
// CHECK-NEXT:   %j0 = arith.constant 0 : index
// CHECK-NEXT:   %v0 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %m0_1 = ptr_xdsl.to_ptr %m0 : memref<2x8xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %pointer_dim_stride = arith.constant 8 : index
// CHECK-NEXT:   %pointer_dim_offset = arith.muli %i0, %pointer_dim_stride : index
// CHECK-NEXT:   %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %j0 : index
// CHECK-NEXT:   %bytes_per_element = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:   %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
// CHECK-NEXT:   %offset_pointer = ptr_xdsl.ptradd %m0_1, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   ptr_xdsl.store %v0, %offset_pointer : vector<8xf32>, !ptr_xdsl.ptr
// CHECK-NEXT: }
