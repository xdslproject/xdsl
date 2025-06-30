// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%lhs, %rhs = "test.op"() : () -> (i32, i64)
%res = "arith.addi"(%lhs, %rhs) : (i32, i64) -> i32
// CHECK: attribute i32 expected from variable 'T', but got i64

// -----

%index = "test.op"() : () -> index
%res = "arith.index_cast"(%index) : (index) -> index
// CHECK: 'arith.index_cast' op operand type 'index' and result type 'index' are cast incompatible

// -----

%i32 = "test.op"() : () -> i32
%res = "arith.index_cast"(%i32) : (i32) -> i32
// CHECK: 'arith.index_cast' op operand type 'i32' and result type 'i32' are cast incompatible

// -----

%c = arith.constant 1 : si32
// CHECK: Expected attribute #builtin.signedness<signless> but got #builtin.signedness<signed>

// -----

%c = arith.constant 1 : ui32
// CHECK: Expected attribute #builtin.signedness<signless> but got #builtin.signedness<unsigned>
