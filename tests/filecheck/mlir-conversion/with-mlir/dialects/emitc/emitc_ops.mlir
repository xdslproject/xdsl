// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

%i32_lhs, %i32_rhs = "test.op"() : () -> (i32, i32)
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
