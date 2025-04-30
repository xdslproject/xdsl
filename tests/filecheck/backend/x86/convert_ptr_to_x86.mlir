// RUN: xdsl-opt -p convert-ptr-to-x86{arch=avx2} --split-input-file  %s | filecheck %s

%ptr = "test.op"(): () -> !ptr_xdsl.ptr
%v = ptr_xdsl.load %ptr : !ptr_xdsl.ptr -> vector<8xf32>
// CHECK:       builtin.module {
// CHECK-NEXT:    %ptr = "test.op"() : () -> !ptr_xdsl.ptr
// CHECK-NEXT:    %v = builtin.unrealized_conversion_cast %ptr : !ptr_xdsl.ptr to !x86.reg
// CHECK-NEXT:    %v_1 = x86.rm.vmovups %v, 0 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT:  }
