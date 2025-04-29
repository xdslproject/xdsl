// RUN: xdsl-opt -p convert-vector-to-ptr %s | filecheck %s

%m = "test.op"(): () -> memref<16xf32>
%i = arith.constant 0: index
%v = vector.load %m[%i]: memref<16xf32>, vector<8xf32>

// CHECK:       builtin.module {
// CHECK-NEXT:    %m = "test.op"() : () -> memref<16xf32>
// CHECK-NEXT:    %i = arith.constant 0 : index
// CHECK-NEXT:    %v = affine.apply affine_map<(d0) -> (d0)> (%i)
// CHECK-NEXT:    %v_1 = ptr_xdsl.to_ptr %m : memref<16xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:    %v_2 = ptr_xdsl.ptradd %v_1, %v : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:    %v_3 = ptr_xdsl.load %v_2 : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK-NEXT:  }
