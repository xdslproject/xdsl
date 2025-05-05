// RUN: xdsl-opt -p convert-vector-to-ptr --split-input-file %s | filecheck %s

%m = "test.op"(): () -> memref<16xf32>
%i = arith.constant 0: index
%v = vector.load %m[%i]: memref<16xf32>, vector<8xf32>

// CHECK:       builtin.module {
// CHECK-NEXT:    %m = "test.op"() : () -> memref<16xf32>
// CHECK-NEXT:    %i = arith.constant 0 : index
// CHECK-NEXT:    %v = affine.apply affine_map<(d0) -> ((d0 * 4))> (%i)
// CHECK-NEXT:    %v_1 = ptr_xdsl.to_ptr %m : memref<16xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:    %v_2 = ptr_xdsl.ptradd %v_1, %v : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:    %v_3 = ptr_xdsl.load %v_2 : !ptr_xdsl.ptr -> vector<8xf32>
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
// CHECK-NEXT:   %v0 = affine.apply affine_map<(d0, d1) -> (((d0 * 32) + (d1 * 4)))> (%i0, %j0)
// CHECK-NEXT:   %v0_1 = ptr_xdsl.to_ptr %m0 : memref<2x8xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v0_2 = ptr_xdsl.ptradd %v0_1, %v0 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v0_3 = ptr_xdsl.load %v0_2 : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK-NEXT: }
