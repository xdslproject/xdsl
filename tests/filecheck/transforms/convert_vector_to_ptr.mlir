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

// -----

%m1 = "test.op"(): () -> memref<2x8xf16>
%i1 = arith.constant 0: index
%j1 = arith.constant 0: index
%v1 = vector.load %m1[%i1,%j1]: memref<2x8xf16>, vector<8xf16>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m1 = "test.op"() : () -> memref<2x8xf16>
// CHECK-NEXT:   %i1 = arith.constant 0 : index
// CHECK-NEXT:   %j1 = arith.constant 0 : index
// CHECK-NEXT:   %v1 = affine.apply affine_map<(d0, d1) -> (((d0 * 16) + (d1 * 2)))> (%i1, %j1)
// CHECK-NEXT:   %v1_1 = ptr_xdsl.to_ptr %m1 : memref<2x8xf16> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v1_2 = ptr_xdsl.ptradd %v1_1, %v1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v1_3 = ptr_xdsl.load %v1_2 : !ptr_xdsl.ptr -> vector<8xf16>
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
// CHECK-NEXT:   %v2 = affine.apply affine_map<(d0, d1) -> (((d0 * 128) + (d1 * 4)))> (%i2, %j2)
// CHECK-NEXT:   %v2_1 = ptr_xdsl.to_ptr %m2 : memref<2x32xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v2_2 = ptr_xdsl.ptradd %v2_1, %v2 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v2_3 = ptr_xdsl.load %v2_2 : !ptr_xdsl.ptr -> vector<8xf32>

// -----

%m3 = "test.op"(): () -> memref<16xf32>
%v3 = "test.op"(): () -> vector<8xf32>
%i3 = arith.constant 0: index
vector.store %v3, %m3[%i3]: memref<16xf32>, vector<8xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m3 = "test.op"() : () -> memref<16xf32>
// CHECK-NEXT:   %v3 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %i3 = arith.constant 0 : index
// CHECK-NEXT:   %0 = affine.apply affine_map<(d0) -> ((d0 * 4))> (%i3)
// CHECK-NEXT:   %1 = ptr_xdsl.to_ptr %m3 : memref<16xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %2 = ptr_xdsl.ptradd %1, %0 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   ptr_xdsl.store %v3, %2 : vector<8xf32>, !ptr_xdsl.ptr
// CHECK-NEXT: }

// -----

%m4 = "test.op"(): () -> memref<2x8xf32>
%v4 = "test.op"(): () -> vector<8xf32>
%i4 = arith.constant 0: index
%j4 = arith.constant 0: index
vector.store %v4, %m4[%i4,%j4]: memref<2x8xf32>, vector<8xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m4 = "test.op"() : () -> memref<2x8xf32>
// CHECK-NEXT:   %v4 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %i4 = arith.constant 0 : index
// CHECK-NEXT:   %j4 = arith.constant 0 : index
// CHECK-NEXT:   %0 = affine.apply affine_map<(d0, d1) -> (((d0 * 32) + (d1 * 4)))> (%i4, %j4)
// CHECK-NEXT:   %1 = ptr_xdsl.to_ptr %m4 : memref<2x8xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %2 = ptr_xdsl.ptradd %1, %0 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   ptr_xdsl.store %v4, %2 : vector<8xf32>, !ptr_xdsl.ptr
// CHECK-NEXT: }

// -----

%m5 = "test.op"(): () -> memref<2x8xf16>
%v5 = "test.op"(): () -> vector<8xf16>
%i5 = arith.constant 0: index
%j5 = arith.constant 0: index
vector.store %v5, %m5[%i5,%j5]: memref<2x8xf16>, vector<8xf16>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m5 = "test.op"() : () -> memref<2x8xf16>
// CHECK-NEXT:   %v5 = "test.op"() : () -> vector<8xf16>
// CHECK-NEXT:   %i5 = arith.constant 0 : index
// CHECK-NEXT:   %j5 = arith.constant 0 : index
// CHECK-NEXT:   %0 = affine.apply affine_map<(d0, d1) -> (((d0 * 16) + (d1 * 2)))> (%i5, %j5)
// CHECK-NEXT:   %1 = ptr_xdsl.to_ptr %m5 : memref<2x8xf16> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %2 = ptr_xdsl.ptradd %1, %0 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   ptr_xdsl.store %v5, %2 : vector<8xf16>, !ptr_xdsl.ptr
// CHECK-NEXT: }

// -----

%m6 = "test.op"(): () -> memref<8xf32, strided<[1], offset: ?>>
%i6 = arith.constant 0: index
%v6 = vector.load %m6[%i6]: memref<8xf32, strided<[1], offset: ?>>, vector<8xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %m6 = "test.op"() : () -> memref<8xf32, strided<[1], offset: ?>>
// CHECK-NEXT:   %i6 = arith.constant 0 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %v6 = affine.apply affine_map<(d0)[s0] -> (((s0 * 4) + (d0 * 4)))> (%i6)[%c0]
// CHECK-NEXT:   %v6_1 = ptr_xdsl.to_ptr %m6 : memref<8xf32, strided<[1], offset: ?>> -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v6_2 = ptr_xdsl.ptradd %v6_1, %v6 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v6_3 = ptr_xdsl.load %v6_2 : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK-NEXT: }
