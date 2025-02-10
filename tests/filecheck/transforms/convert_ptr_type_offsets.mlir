// RUN: xdsl-opt -p convert-ptr-type-offsets --split-input-file --verify-diagnostics %s | filecheck %s

%a1 = ptr_xdsl.type_offset i32 : index
// CHECK: %a1 = arith.constant 4 : index

%a2 = ptr_xdsl.type_offset f128 : index
// CHECK-NEXT: %a2 = arith.constant 16 : index

// -----

%a3 = ptr_xdsl.type_offset tensor<4xi32> : index
// CHECK: Type offset is currently only supported for fixed size types
