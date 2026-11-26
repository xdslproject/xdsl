// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

%i32_lhs, %i32_rhs = "test.op"() : () -> (i32, i32)
%f32_lhs, %f32_rhs = "test.op"() : () -> (f32, f32)
%size_lhs, %size_rhs = "test.op"() : () -> (!emitc.size_t, !emitc.size_t)
%ptr_f32, %i32_offset = "test.op"() : () -> (!emitc.ptr<f32>, i32)
%opaque_uint = "test.op"() : () -> !emitc.opaque<"unsigned int">
%tensor_lhs, %tensor_rhs = "test.op"() : () -> (tensor<3x4xi32>, tensor<3x4xi32>)
%lvalue_i32 = "test.op"() : () -> !emitc.lvalue<i32>
%lvalue_f32 = "test.op"() : () -> !emitc.lvalue<f32>
%ptr_i32 = "test.op"() : () -> !emitc.ptr<i32>
%ptr_ptr_i32 = "test.op"() : () -> !emitc.ptr<!emitc.ptr<i32>>
%lvalue_opaque = "test.op"() : () -> !emitc.lvalue<!emitc.opaque<"MyType">>

//===----------------------------------------------------------------------===//
// CallOpaqueOp
//===----------------------------------------------------------------------===//

%0 = emitc.call_opaque "blah"() : () -> i64
emitc.call_opaque "foo" (%0) {args = [
  0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
]} : (i64) -> ()
emitc.call_opaque "test" ()  : () -> ()

// CHECK: {{%.*}} = emitc.call_opaque "blah"() : () -> i64
// CHECK-NEXT: emitc.call_opaque "foo"({{%.*}}) {args = [0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index]} : (i64) -> ()
// CHECK-NEXT: emitc.call_opaque "test"() : () -> ()

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

%add_int = emitc.add %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.add {{%.*}}, {{%.*}} : (i32, i32) -> i32

%add_ptr_int = emitc.add %ptr_f32, %i32_offset : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
// CHECK: {{%.*}} = emitc.add {{%.*}}, {{%.*}} : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>

%add_ptr_opaque = emitc.add %ptr_f32, %opaque_uint : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
// CHECK: {{%.*}} = emitc.add {{%.*}}, {{%.*}} : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>

%add_tensor = emitc.add %tensor_lhs, %tensor_rhs : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
// CHECK: {{%.*}} = emitc.add {{%.*}}, {{%.*}} : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

// Test addressof operator with generic syntax
%apply_addressof_generic = "emitc.apply"(%lvalue_i32) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK: {{%.*}} = emitc.apply "&"({{%.*}}) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

// Test addressof operator with custom syntax
%apply_addressof_custom = emitc.apply "&"(%lvalue_i32) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK: {{%.*}} = emitc.apply "&"({{%.*}}) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

// Test dereference operator with generic syntax
%apply_dereference_generic = "emitc.apply"(%ptr_i32) {applicableOperator = "*"} : (!emitc.ptr<i32>) -> i32
// CHECK: {{%.*}} = emitc.apply "*"({{%.*}}) : (!emitc.ptr<i32>) -> i32

// Test dereference operator with custom syntax
%apply_dereference_custom = emitc.apply "*"(%ptr_i32) : (!emitc.ptr<i32>) -> i32
// CHECK: {{%.*}} = emitc.apply "*"({{%.*}}) : (!emitc.ptr<i32>) -> i32

// Test addressof with f32 type
%apply_addressof_f32 = emitc.apply "&"(%lvalue_f32) : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
// CHECK: {{%.*}} = emitc.apply "&"({{%.*}}) : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>

// Test dereference with f32 type
%apply_dereference_f32 = emitc.apply "*"(%ptr_f32) : (!emitc.ptr<f32>) -> f32
// CHECK: {{%.*}} = emitc.apply "*"({{%.*}}) : (!emitc.ptr<f32>) -> f32

// Test nested pointer operations
%apply_nested_deref1 = emitc.apply "*"(%ptr_ptr_i32) : (!emitc.ptr<!emitc.ptr<i32>>) -> !emitc.ptr<i32>
// CHECK: {{%.*}} = emitc.apply "*"({{%.*}}) : (!emitc.ptr<!emitc.ptr<i32>>) -> !emitc.ptr<i32>

%apply_nested_deref2 = emitc.apply "*"(%apply_nested_deref1) : (!emitc.ptr<i32>) -> i32
// CHECK: {{%.*}} = emitc.apply "*"({{%.*}}) : (!emitc.ptr<i32>) -> i32

// Test with opaque type
%apply_opaque = emitc.apply "&"(%lvalue_opaque) : (!emitc.lvalue<!emitc.opaque<"MyType">>) -> !emitc.ptr<!emitc.opaque<"MyType">>
// CHECK: {{%.*}} = emitc.apply "&"({{%.*}}) : (!emitc.lvalue<!emitc.opaque<"MyType">>) -> !emitc.ptr<!emitc.opaque<"MyType">>

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

emitc.assign %i32_lhs : i32 to %lvalue_i32 : !emitc.lvalue<i32>
// CHECK: emitc.assign {{%.*}} : i32 to {{%.*}} : <i32>

//===----------------------------------------------------------------------===//
// BitwiseLeftShiftOp
//===----------------------------------------------------------------------===//

%bitwise_left_shift = emitc.bitwise_left_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.bitwise_left_shift {{%.*}}, {{%.*}} : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseNotOp
//===----------------------------------------------------------------------===//

%bitwise_not = emitc.bitwise_not %i32_lhs : (i32) -> i32
// CHECK: {{%.*}} = emitc.bitwise_not {{%.*}} : (i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseOrOp
//===----------------------------------------------------------------------===//

%bitwise_or = emitc.bitwise_or %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.bitwise_or {{%.*}}, {{%.*}} : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseRightShiftOp
//===----------------------------------------------------------------------===//

%bitwise_right_shift = emitc.bitwise_right_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.bitwise_right_shift {{%.*}}, {{%.*}} : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseXorOp
//===----------------------------------------------------------------------===//

%bitwise_xor = emitc.bitwise_xor %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.bitwise_xor {{%.*}}, {{%.*}} : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

%cons_int = "emitc.constant"() <{value = 42 : i32}> : () -> i32
// CHECK: {{%.*}} = "emitc.constant"() <{value = 42 : i32}> : () -> i32

%cons_float = "emitc.constant"() <{value = 1.500000e+00 : f32}> : () -> f32
// CHECK: {{%.*}} = "emitc.constant"() <{value = 1.500000e+00 : f32}> : () -> f32

%cons_sizet = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.size_t
// CHECK: {{%.*}} = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.size_t

%cons_ssizet = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ssize_t
// CHECK: {{%.*}} = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ssize_t

%cons_ptrdiff = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ptrdiff_t
// CHECK: {{%.*}} = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ptrdiff_t

%cons_opaque = "emitc.constant"() <{value = #emitc.opaque<"42">}> : () -> !emitc.opaque<"int">
// CHECK: {{%.*}} = "emitc.constant"() <{value = #emitc.opaque<"42">}> : () -> !emitc.opaque<"int">

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

%div_int = emitc.div %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.div {{%.*}}, {{%.*}} : (i32, i32) -> i32

%div_float = emitc.div %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK: {{%.*}} = emitc.div {{%.*}}, {{%.*}} : (f32, f32) -> f32

%div_size = emitc.div %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK: {{%.*}} = emitc.div {{%.*}}, {{%.*}} : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t

//===----------------------------------------------------------------------===//
// LiteralOp
//===----------------------------------------------------------------------===//

%lit = emitc.literal "1" : i32
// CHECK: {{%.*}} = emitc.literal "1" : i32

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

%load = emitc.load %lvalue_i32 : !emitc.lvalue<i32>
// CHECK: {{%.*}} = emitc.load {{%.*}} : <i32>

//===----------------------------------------------------------------------===//
// LogicalAndOp
//===----------------------------------------------------------------------===//

%logical_and = emitc.logical_and %i32_lhs, %i32_rhs : i32, i32
// CHECK: {{%.*}} = emitc.logical_and {{%.*}}, {{%.*}} : i32, i32

//===----------------------------------------------------------------------===//
// LogicalNotOp
//===----------------------------------------------------------------------===//

%logical_not = emitc.logical_not %i32_lhs : i32
// CHECK: {{%.*}} = emitc.logical_not {{%.*}} : i32

//===----------------------------------------------------------------------===//
// LogicalOrOp
//===----------------------------------------------------------------------===//

%logical_or = emitc.logical_or %i32_lhs, %i32_rhs : i32, i32
// CHECK: {{%.*}} = emitc.logical_or {{%.*}}, {{%.*}} : i32, i32

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

%mul_int = emitc.mul %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.mul {{%.*}}, {{%.*}} : (i32, i32) -> i32

%mul_float = emitc.mul %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK: {{%.*}} = emitc.mul {{%.*}}, {{%.*}} : (f32, f32) -> f32

%mul_size = emitc.mul %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK: {{%.*}} = emitc.mul {{%.*}}, {{%.*}} : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t

//===----------------------------------------------------------------------===//
// RemOp
//===----------------------------------------------------------------------===//

%rem_int = emitc.rem %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.rem {{%.*}}, {{%.*}} : (i32, i32) -> i32

%rem_size = emitc.rem %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK: {{%.*}} = emitc.rem {{%.*}}, {{%.*}} : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

%sub_int = emitc.sub %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: {{%.*}} = emitc.sub {{%.*}}, {{%.*}} : (i32, i32) -> i32

%sub_ptr_int = emitc.sub %ptr_f32, %i32_offset : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
// CHECK: {{%.*}} = emitc.sub {{%.*}}, {{%.*}} : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>

%sub_ptr_opaque = emitc.sub %ptr_f32, %opaque_uint : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
// CHECK: {{%.*}} = emitc.sub {{%.*}}, {{%.*}} : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>

%sub_ptr_ptr = emitc.sub %ptr_i32, %ptr_i32 : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> !emitc.ptrdiff_t
// CHECK: {{%.*}} = emitc.sub {{%.*}}, {{%.*}} : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> !emitc.ptrdiff_t

%sub_tensor = emitc.sub %tensor_lhs, %tensor_rhs : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
// CHECK: {{%.*}} = emitc.sub {{%.*}}, {{%.*}} : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>

//===----------------------------------------------------------------------===//
// UnaryMinusOp
//===----------------------------------------------------------------------===//

%unary_minus = emitc.unary_minus %i32_lhs : (i32) -> i32
// CHECK: {{%.*}} = emitc.unary_minus {{%.*}} : (i32) -> i32

//===----------------------------------------------------------------------===//
// UnaryPlusOp
//===----------------------------------------------------------------------===//

%unary_plus = emitc.unary_plus %i32_lhs : (i32) -> i32
// CHECK: {{%.*}} = emitc.unary_plus {{%.*}} : (i32) -> i32

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

%variable = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK: {{%.*}} = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>

%variable_init = "emitc.variable"() <{value = 7 : i32}> : () -> !emitc.lvalue<i32>
// CHECK: {{%.*}} = "emitc.variable"() <{value = 7 : i32}> : () -> !emitc.lvalue<i32>

%variable_sizet = "emitc.variable"() <{value = 42 : index}> : () -> !emitc.lvalue<!emitc.size_t>
// CHECK: {{%.*}} = "emitc.variable"() <{value = 42 : index}> : () -> !emitc.lvalue<!emitc.size_t>

%variable_ptr = "emitc.variable"() <{value = #emitc.opaque<"NULL">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>
// CHECK: {{%.*}} = "emitc.variable"() <{value = #emitc.opaque<"NULL">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>

%variable_array = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<2xi32>
// CHECK: {{%.*}} = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<2xi32>
