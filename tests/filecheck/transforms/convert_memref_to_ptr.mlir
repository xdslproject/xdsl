%v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
memref.store %v, %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:    %arr_ptr  = memref.to_ptr %arr : !ptr.ptr
// CHECK-NEXT:    %type_offset = ptr.type_offset i32 : index
// CHECK-NEXT:    %offset = index.mul %type_offset, %idx : index
// CHECK-NEXT:    %element_ptr = ptr.ptradd %arr_ptr, %offset : !ptr.ptr
// CHECK-NEXT:    ptr.store %v, %element_ptr : i32

%idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
memref.store %v, %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:    %arr2_ptr  = memref.to_ptr %arr_2 : !ptr.ptr
// CHECK-NEXT:    %type_offset = ptr.type_offset i32 : index
// CHECK-NEXT:    %pointer_dim_stride = index.casts 10 : i32 to index
// CHECK-NEXT:    %pointer_dim_offset = index.mul %pointer_dim_stride, %idx1 : index
// CHECK-NEXT:    %scaled_dim_offset = index.mul %pointer_dim_offset, %type_offset : index
// CHECK-NEXT:    %second_dim_offset = index.mul %idx2, %type_offset : index
// CHECK-NEXT:    %offset = index.add %scaled_dim_offset, %second_dim_offset : index
// CHECK-NEXT:    %element_ptr = ptr.ptradd %arr2_ptr, %offset : !ptr.ptr
// CHECK-NEXT:    ptr.store %v, %element_ptr : i32
