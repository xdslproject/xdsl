// RUN: xdsl-opt %s -p convert-ptr-to-riscv --split-input-file | filecheck %s

// CHECK: builtin.module {
builtin.module {

  %m, %idx, %v = "test.op"() : () -> (memref<3x2xi32>, index, i32)
  %p = "memref.to_ptr"(%m) : (memref<3x2xi32>) -> !ptr.ptr

  // CHECK-NEXT: %m, %idx, %v = "test.op"() : () -> (memref<3x2xi32>, index, i32)
  // CHECK-NEXT: %p = builtin.unrealized_conversion_cast %m : memref<3x2xi32> to index

  %r0 = ptr.ptradd %p, %idx : (!ptr.ptr, index) -> !ptr.ptr
  
  // CHECK-NEXT: %r0 = arith.addi %p, %idx : index

  %r1 = "ptr.type_offset"() <{"elem_type" = i32}> : () -> index
  
  // CHECK-NEXT: %r1 = arith.constant 4 : index

  ptr.store %v, %p : i32, !ptr.ptr
  
  // CHECK-NEXT:  %p_1 = builtin.unrealized_conversion_cast %p : index to !riscv.reg
  // CHECK-NEXT:  %v_1 = builtin.unrealized_conversion_cast %v : i32 to !riscv.reg
  // CHECK-NEXT:  riscv.sw %p_1, %v_1, 0 {"comment" = "store int value to pointer"} : (!riscv.reg, !riscv.reg) -> ()

  %r3 = ptr.load %p : !ptr.ptr -> i32

  // CHECK-NEXT:  %p_2 = builtin.unrealized_conversion_cast %p : index to !riscv.reg
  // CHECK-NEXT:  %r3 = riscv.lw %p_2, 0 {"comment" = "load word from pointer"} : (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT:  %r3_1 = builtin.unrealized_conversion_cast %r3 : !riscv.reg to i32
}

// CHECK-NEXT: }
