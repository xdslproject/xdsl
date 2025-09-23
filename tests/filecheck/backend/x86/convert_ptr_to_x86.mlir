// RUN: xdsl-opt -p convert-ptr-to-x86{arch=avx512} --verify-diagnostics --split-input-file  %s | filecheck %s

%ptr0 = "test.op"(): () -> !ptr_xdsl.ptr
%v0 = ptr_xdsl.load %ptr0 : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK:       builtin.module {
// CHECK-NEXT:    %ptr0 = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:    %ptr0_1 = builtin.unrealized_conversion_cast %ptr0 : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:    %v0 = x86.dm.vmovups %ptr0_1, 0 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT:    %v0_1 = builtin.unrealized_conversion_cast %v0 : !x86.avx2reg to vector<8xf32>
// CHECK-NEXT:  }

// -----

%ptr0b = "test.op"(): () -> !ptr_xdsl.ptr
%v0b = ptr_xdsl.load %ptr0b : !ptr_xdsl.ptr -> vector<16xf32>
// CHECK:       builtin.module {
// CHECK-NEXT:    %ptr0b = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:    %ptr0b_1 = builtin.unrealized_conversion_cast %ptr0b : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:    %v0b = x86.dm.vmovups %ptr0b_1, 0 : (!x86.reg) -> !x86.avx512reg
// CHECK-NEXT:    %v0b_1 = builtin.unrealized_conversion_cast %v0b : !x86.avx512reg to vector<16xf32>
// CHECK-NEXT:  }

// -----

%ptr1 = "test.op"(): () -> !ptr_xdsl.ptr
%v1 = ptr_xdsl.load %ptr1 : !ptr_xdsl.ptr -> f32
// CHECK:      builtin.module {
// CHECK-NEXT:   %ptr1 = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:   %ptr1_1 = builtin.unrealized_conversion_cast %ptr1 : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:   %v1 = x86.dm.mov %ptr1_1, 0 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:   %v1_1 = builtin.unrealized_conversion_cast %v1 : !x86.reg to f32
// CHECK-NEXT: }

// -----

// CHECK: Half-precision floating point vector load is not implemented yet.
%ptr2 = "test.op"(): () -> !ptr_xdsl.ptr
%v2 = ptr_xdsl.load %ptr2 : !ptr_xdsl.ptr -> vector<8xf16>

// -----

%ptr3 = "test.op"(): () -> !ptr_xdsl.ptr
%v3 = ptr_xdsl.load %ptr3 : !ptr_xdsl.ptr -> vector<4xf64>

// CHECK:      builtin.module {
// CHECK-NEXT:   %ptr3 = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:   %ptr3_1 = builtin.unrealized_conversion_cast %ptr3 : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:   %v3 = x86.dm.vmovupd %ptr3_1, 0 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT:   %v3_1 = builtin.unrealized_conversion_cast %v3 : !x86.avx2reg to vector<4xf64>
// CHECK-NEXT: }

// -----

// CHECK: The vector size (1024 bits) and target architecture `avx512` are inconsistent.
%ptr4 = "test.op"(): () -> !ptr_xdsl.ptr
%v4 = ptr_xdsl.load %ptr4 : !ptr_xdsl.ptr -> vector<32xf32>

// -----

// CHECK: Float precision must be half, single or double.
%ptr5 = "test.op"(): () -> !ptr_xdsl.ptr
%v5 = ptr_xdsl.load %ptr5 : !ptr_xdsl.ptr -> vector<1xf128>

// -----

%ptr6 = "test.op"(): () -> !ptr_xdsl.ptr
%v6 = "test.op"(): () -> vector<8xf32>
ptr_xdsl.store %v6, %ptr6 : vector<8xf32>, !ptr_xdsl.ptr

// CHECK:      builtin.module {
// CHECK-NEXT:   %ptr6 = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v6 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %ptr6_1 = builtin.unrealized_conversion_cast %ptr6 : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:   %v6_1 = builtin.unrealized_conversion_cast %v6 : vector<8xf32> to !x86.avx2reg
// CHECK-NEXT:   x86.ms.vmovups %ptr6_1, %v6_1, 0 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: }

// -----

%ptr6b = "test.op"(): () -> !ptr_xdsl.ptr
%v6b = "test.op"(): () -> vector<16xf32>
ptr_xdsl.store %v6b, %ptr6b : vector<16xf32>, !ptr_xdsl.ptr

// CHECK:      builtin.module {
// CHECK-NEXT:   %ptr6b = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v6b = "test.op"() : () -> vector<16xf32>
// CHECK-NEXT:   %ptr6b_1 = builtin.unrealized_conversion_cast %ptr6b : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:   %v6b_1 = builtin.unrealized_conversion_cast %v6b : vector<16xf32> to !x86.avx512reg
// CHECK-NEXT:   x86.ms.vmovups %ptr6b_1, %v6b_1, 0 : (!x86.reg, !x86.avx512reg) -> ()
// CHECK-NEXT: }

// -----

%ptr6 = "test.op"(): () -> !ptr_xdsl.ptr
%v6 = "test.op"(): () -> vector<4xf64>
ptr_xdsl.store %v6, %ptr6 : vector<4xf64>, !ptr_xdsl.ptr

// CHECK:      builtin.module {
// CHECK-NEXT:   %ptr6 = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v6 = "test.op"() : () -> vector<4xf64>
// CHECK-NEXT:   %ptr6_1 = builtin.unrealized_conversion_cast %ptr6 : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:   %v6_1 = builtin.unrealized_conversion_cast %v6 : vector<4xf64> to !x86.avx2reg
// CHECK-NEXT:   x86.ms.vmovapd %ptr6_1, %v6_1, 0 : (!x86.reg, !x86.avx2reg) -> ()
// CHECK-NEXT: }

// -----

// CHECK: Half-precision floating point vector load is not implemented yet.
%ptr6 = "test.op"(): () -> !ptr_xdsl.ptr
%v6 = "test.op"(): () -> vector<16xf16>
ptr_xdsl.store %v6, %ptr6 : vector<16xf16>, !ptr_xdsl.ptr

// -----

// CHECK: Float precision must be half, single or double.
%ptr6 = "test.op"(): () -> !ptr_xdsl.ptr
%v6 = "test.op"(): () -> vector<1xf128>
ptr_xdsl.store %v6, %ptr6 : vector<1xf128>, !ptr_xdsl.ptr

// -----

%ptr6 = "test.op"(): () -> !ptr_xdsl.ptr
%v6 = "test.op"(): () -> f32
ptr_xdsl.store %v6, %ptr6 : f32, !ptr_xdsl.ptr

// CHECK:      builtin.module {
// CHECK-NEXT:   %ptr6 = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:   %v6 = "test.op"() : () -> f32
// CHECK-NEXT:   %ptr6_1 = builtin.unrealized_conversion_cast %ptr6 : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:   %v6_1 = builtin.unrealized_conversion_cast %v6 : f32 to !x86.reg
// CHECK-NEXT:   x86.ms.mov %ptr6_1, %v6_1, 0 : (!x86.reg, !x86.reg) -> ()
// CHECK-NEXT: }

// -----

%p = "test.op"(): () -> !ptr_xdsl.ptr
%idx = "test.op"(): () -> index
%r0 = ptr_xdsl.ptradd %p, %idx : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr

// CHECK:      builtin.module {
// CHECK-NEXT:   %p = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:   %idx = "test.op"() : () -> index
// CHECK-NEXT:   %r0 = builtin.unrealized_conversion_cast %p : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:   %r0_1 = builtin.unrealized_conversion_cast %idx : index to !x86.reg
// CHECK-NEXT:   %r0_2 = x86.ds.mov %r0 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:   %r0_3 = x86.rs.add %r0_2, %r0_1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT:   %r0_4 = builtin.unrealized_conversion_cast %r0_3 : !x86.reg to !ptr_xdsl.ptr
// CHECK-NEXT: }
