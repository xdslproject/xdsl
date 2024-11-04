// RUN: xdsl-opt %s -p convert-ptr-to-riscv --split-input-file | filecheck %s

%m, %idx, %v = "test.op"() : () -> (memref<3x2xi32>, index, i32)

%p = ptr_xdsl.to_ptr %m : memref<3x2xi32> -> !ptr_xdsl.ptr
// CHECK: %p = builtin.unrealized_conversion_cast %m : memref<3x2xi32> to index

%r0 = ptr_xdsl.ptradd %p, %idx : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %r0 = arith.addi %p, %idx : index

ptr_xdsl.store %v, %p : i32, !ptr_xdsl.ptr
// CHECK-NEXT:  %p_1 = builtin.unrealized_conversion_cast %p : index to !riscv.reg
// CHECK-NEXT:  %v_1 = builtin.unrealized_conversion_cast %v : i32 to !riscv.reg
// CHECK-NEXT:  riscv.sw %p_1, %v_1, 0 {"comment" = "store int value to pointer"} : (!riscv.reg, !riscv.reg) -> ()

%r3 = ptr_xdsl.load %p : !ptr_xdsl.ptr -> i32
// CHECK-NEXT:  %p_2 = builtin.unrealized_conversion_cast %p : index to !riscv.reg
// CHECK-NEXT:  %r3 = riscv.lw %p_2, 0 {"comment" = "load word from pointer"} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %r3_1 = builtin.unrealized_conversion_cast %r3 : !riscv.reg to i32
