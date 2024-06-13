// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %lhs, %rhs = "test.op"() : () -> (i32, i64)
  %res = "arith.addi"(%lhs, %rhs) : (i32, i64) -> i32

  // CHECK: attribute i32 expected from variable 'T', but got i64
}) : () -> ()

// -----

"builtin.module"() ({

  %index = "test.op"() : () -> index
  %res = "arith.index_cast"(%index) : (index) -> index
  // CHECK: 'arith.index_cast' op operand type 'index' and result type 'index' are cast incompatible

}) : () -> ()

// -----

"builtin.module"() ({

  %i32 = "test.op"() : () -> i32
  %res = "arith.index_cast"(%i32) : (i32) -> i32
  // CHECK: 'arith.index_cast' op operand type 'i32' and result type 'i32' are cast incompatible

}) : () -> ()
