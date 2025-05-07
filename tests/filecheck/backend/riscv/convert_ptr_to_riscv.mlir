// RUN: xdsl-opt %s -p convert-ptr-to-riscv --split-input-file --verify-diagnostics | filecheck %s

%m, %idx, %v = "test.op"() : () -> (memref<3x2xi32>, index, i32)

%p = ptr_xdsl.to_ptr %m : memref<3x2xi32> -> !ptr_xdsl.ptr
// CHECK: %p = builtin.unrealized_conversion_cast %m : memref<3x2xi32> to !riscv.reg

%r0 = ptr_xdsl.ptradd %p, %idx : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %idx_1 = builtin.unrealized_conversion_cast %idx : index to !riscv.reg
// CHECK-NEXT:  %r0 = riscv.add %p, %idx_1 : (!riscv.reg, !riscv.reg) -> !riscv.reg

ptr_xdsl.store %v, %p : i32, !ptr_xdsl.ptr
// CHECK-NEXT:  %v_1 = builtin.unrealized_conversion_cast %v : i32 to !riscv.reg
// CHECK-NEXT:  riscv.sw %p, %v_1, 0 {comment = "store int value to pointer"} : (!riscv.reg, !riscv.reg) -> ()

%r3 = ptr_xdsl.load %p : !ptr_xdsl.ptr -> i32
// CHECK-NEXT:  %r3 = riscv.lw %p, 0 {comment = "load word from pointer"} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %r3_1 = builtin.unrealized_conversion_cast %r3 : !riscv.reg to i32

// -----

%m2 = "test.op"() : () -> (memref<3x2xf128>)
%p2 = ptr_xdsl.to_ptr %m2 : memref<3x2xf128> -> !ptr_xdsl.ptr
%v1 = ptr_xdsl.load %p2 : !ptr_xdsl.ptr -> f128
// CHECK: Lowering memref.load op with floating point type f128 not yet implemented
