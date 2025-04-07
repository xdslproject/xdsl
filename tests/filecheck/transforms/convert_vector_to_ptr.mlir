// RUN: xdsl-opt -p convert-vector-to-ptr %s | filecheck %s

%m = "test.op"(): () -> memref<16xf32>
%i = arith.constant 0: index
%v = vector.load %m[%i]: memref<16xf32>, vector<8xf32>

// CHECK:       builtin.module {
// CHECK-NEXT:    %m = "test.op"() : () -> memref<16xf32>
// CHECK-NEXT:    %i = arith.constant 0 : index
// CHECK-NEXT:    %v = memref.subview %m[%i] [8] [1] : memref<16xf32> to memref<8xf32>
// CHECK-NEXT:    %v_1 = builtin.unrealized_conversion_cast %v : memref<8xf32> to !ptr_xdsl.ptr
// CHECK-NEXT:    %v_2 = ptr_xdsl.load %v_1 : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK-NEXT:  }
