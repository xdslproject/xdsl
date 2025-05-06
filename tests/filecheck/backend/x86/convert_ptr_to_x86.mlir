// RUN: xdsl-opt -p convert-ptr-to-x86{arch=avx2} --verify-diagnostics --split-input-file  %s | filecheck %s

%ptr0 = "test.op"(): () -> !ptr_xdsl.ptr
%v0 = ptr_xdsl.load %ptr0 : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK:       builtin.module {
// CHECK-NEXT:    %ptr0 = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:    %v0 = builtin.unrealized_conversion_cast %ptr0 : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:    %v0_1 = x86.rm.vmovups %v0, 0 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT:  }


// -----

// CHECK: The lowering of ptr.load is not yet implemented for non-vector types.
%ptr1 = "test.op"(): () -> !ptr_xdsl.ptr
%v1 = ptr_xdsl.load %ptr1 : !ptr_xdsl.ptr -> f32

// -----

// CHECK: Half-precision vector load is not implemented yet.
%ptr2 = "test.op"(): () -> !ptr_xdsl.ptr
%v2 = ptr_xdsl.load %ptr2 : !ptr_xdsl.ptr -> vector<8xf16>

// -----

// CHECK: Double precision vector load is not implemented yet.
%ptr3 = "test.op"(): () -> !ptr_xdsl.ptr
%v3 = ptr_xdsl.load %ptr3 : !ptr_xdsl.ptr -> vector<4xf64>

// -----

// CHECK: The vector size and target architecture are inconsistent.
%ptr4 = "test.op"(): () -> !ptr_xdsl.ptr
%v4 = ptr_xdsl.load %ptr4 : !ptr_xdsl.ptr -> vector<8xf64>

// -----

// CHECK: Float precision must be half, single or double.
%ptr5 = "test.op"(): () -> !ptr_xdsl.ptr
%v5 = ptr_xdsl.load %ptr5 : !ptr_xdsl.ptr -> vector<1xf128>
