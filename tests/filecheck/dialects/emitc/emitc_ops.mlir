// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%i32_lhs, %i32_rhs = "test.op"() : () -> (i32, i32)
%f32_lhs, %f32_rhs = "test.op"() : () -> (f32, f32)
%ptr_f32, %i32_offset = "test.op"() : () -> (!emitc.ptr<f32>, i32)
%opaque_uint = "test.op"() : () -> !emitc.opaque<"unsigned int">
%tensor_lhs, %tensor_rhs = "test.op"() : () -> (tensor<3x4xi32>, tensor<3x4xi32>)

//===----------------------------------------------------------------------===//
// CallOpaqueOp
//===----------------------------------------------------------------------===//

%0 = emitc.call_opaque "blah"() : () -> i64
emitc.call_opaque "foo" (%0) {args = [
  0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
]} : (i64) -> ()
emitc.call_opaque "test" ()  : () -> ()

// CHECK:  %0 = emitc.call_opaque "blah"() : () -> i64
// CHECK-NEXT:  emitc.call_opaque "foo"(%0) {args = [0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index]} : (i64) -> ()
// CHECK-NEXT:  emitc.call_opaque "test"() : () -> ()

// CHECK-GENERIC:  %0 = "emitc.call_opaque"() <{callee = "blah"}> : () -> i64
// CHECK-GENERIC-NEXT: "emitc.call_opaque"(%0) <{callee = "foo", args = [0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index]}> : (i64) -> ()
// CHECK-GENERIC-NEXT: "emitc.call_opaque"() <{callee = "test"}> : () -> ()

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

%add_int = emitc.add %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %add_int = emitc.add %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %add_int = "emitc.add"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

%add_ptr_int = emitc.add %ptr_f32, %i32_offset : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
// CHECK: %add_ptr_int = emitc.add %ptr_f32, %i32_offset : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
// CHECK-GENERIC: %add_ptr_int = "emitc.add"(%ptr_f32, %i32_offset) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>

%add_ptr_opaque = emitc.add %ptr_f32, %opaque_uint : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
// CHECK: %add_ptr_opaque = emitc.add %ptr_f32, %opaque_uint : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
// CHECK-GENERIC: %add_ptr_opaque = "emitc.add"(%ptr_f32, %opaque_uint) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>

%add_tensor = emitc.add %tensor_lhs, %tensor_rhs : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
// CHECK: %add_tensor = emitc.add %tensor_lhs, %tensor_rhs : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
// CHECK-GENERIC: %add_tensor = "emitc.add"(%tensor_lhs, %tensor_rhs) : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

%cons_int = "emitc.constant"() {value = 42 : i32} : () -> i32
// CHECK %cons_int = "emitc.constant"() {value = 42 : i32} : () -> i32
// CHECK-GENERIC %cons_int = "emitc.constant"() {value = 42 : i32} : () -> i32

%cons_sizet = "emitc.constant"(){value = 42 : index} : () -> !emitc.size_t
// CHECK %cons_ind = "emitc.constant"(){value = 42 : index} : () -> !emitc.size_t
// CHECK-GENERIC %cons_ind = "emitc.constant"(){value = 42 : index} : () -> !emitc.size_t

%cons_ssizet = "emitc.constant"(){value = 42 : index} : () -> !emitc.ssize_t
// CHECK %cons_ssizet = "emitc.constant"(){value = 42 : index} : () -> !emitc.ssize_t
// CHECK-GENERIC %cons_ssizet = "emitc.constant"(){value = 42 : index} : () -> !emitc.ssize_t

%cons_ptr = "emitc.constant"(){value = 42 : index} : () -> !emitc.ptrdiff_t
// CHECK %cons_ptr = "emitc.constant"(){value = 42 : index} : () -> !emitc.ptrdiff_t
// CHECK-GENERIC %cons_ptr = "emitc.constant"(){value = 42 : index} : () -> !emitc.ptrdiff_t

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

%variable = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK: %variable = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK-GENERIC: %variable = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

emitc.assign %cons_int : i32 to %variable : !emitc.lvalue<i32>
// CHECK: emitc.assign %cons_int : i32 to %variable : !emitc.lvalue<i32>

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

%sub_int = emitc.sub %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %sub_int = emitc.sub %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %sub_int = "emitc.sub"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

%sub_ptr_int = emitc.sub %ptr_f32, %i32_offset : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
// CHECK: %sub_ptr_int = emitc.sub %ptr_f32, %i32_offset : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
// CHECK-GENERIC: %sub_ptr_int = "emitc.sub"(%ptr_f32, %i32_offset) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>

%sub_ptr_opaque = emitc.sub %ptr_f32, %opaque_uint : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
// CHECK: %sub_ptr_opaque = emitc.sub %ptr_f32, %opaque_uint : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
// CHECK-GENERIC: %sub_ptr_opaque = "emitc.sub"(%ptr_f32, %opaque_uint) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>

%sub_tensor = emitc.sub %tensor_lhs, %tensor_rhs : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
// CHECK: %sub_tensor = emitc.sub %tensor_lhs, %tensor_rhs : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
// CHECK-GENERIC: %sub_tensor = "emitc.sub"(%tensor_lhs, %tensor_rhs) : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

%mul_int = emitc.mul %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %mul_int = emitc.mul %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %mul_int = "emitc.mul"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

%mul_float = emitc.mul %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK: %mul_float = emitc.mul %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK-GENERIC: %mul_float = "emitc.mul"(%f32_lhs, %f32_rhs) : (f32, f32) -> f32

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

%div_int = emitc.div %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %div_int = emitc.div %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %div_int = "emitc.div"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

%div_float = emitc.div %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK: %div_float = emitc.div %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK-GENERIC: %div_float = "emitc.div"(%f32_lhs, %f32_rhs) : (f32, f32) -> f32

//===----------------------------------------------------------------------===//
// RemOp
//===----------------------------------------------------------------------===//

%rem_int = emitc.rem %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %rem_int = emitc.rem %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %rem_int = "emitc.rem"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseAndOp
//===----------------------------------------------------------------------===//

%and_int = emitc.bitwise_and %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %and_int = emitc.bitwise_and %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %and_int = "emitc.bitwise_and"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseLeftShiftOp
//===----------------------------------------------------------------------===//

%left_shift_int = emitc.bitwise_left_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %left_shift_int = emitc.bitwise_left_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %left_shift_int = "emitc.bitwise_left_shift"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseNotOp
//===----------------------------------------------------------------------===//

%not_int = emitc.bitwise_not %i32_lhs : (i32) -> i32
// CHECK: %not_int = emitc.bitwise_not %i32_lhs : (i32) -> i32
// CHECK-GENERIC: %not_int = "emitc.bitwise_not"(%i32_lhs) : (i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseOrOp
//===----------------------------------------------------------------------===//

%or_int = emitc.bitwise_or %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %or_int = emitc.bitwise_or %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %or_int = "emitc.bitwise_or"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseRightShiftOp
//===----------------------------------------------------------------------===//

%right_shift_int = emitc.bitwise_right_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %right_shift_int = emitc.bitwise_right_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %right_shift_int = "emitc.bitwise_right_shift"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseXorOp
//===----------------------------------------------------------------------===//

%xor_int = emitc.bitwise_xor %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %xor_int = emitc.bitwise_xor %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %xor_int = "emitc.bitwise_xor"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// LogicalAndOp
//===----------------------------------------------------------------------===//

%log_and = emitc.logical_and %i32_lhs, %i32_rhs : i32, i32
// CHECK: %log_and = emitc.logical_and %i32_lhs, %i32_rhs : i32, i32
// CHECK-GENERIC: %log_and = "emitc.logical_and"(%i32_lhs, %i32_rhs) : (i32, i32)

//===----------------------------------------------------------------------===//
// LogicalNotOp
//===----------------------------------------------------------------------===//

%log_not = emitc.logical_not %i32_lhs : i32
// CHECK: %log_not = emitc.logical_not %i32_lhs : i32
// CHECK-GENERIC: %log_not = "emitc.logical_not"(%i32_lhs) : (i32)

//===----------------------------------------------------------------------===//
// LogicalOrOp
//===----------------------------------------------------------------------===//

%log_or = emitc.logical_or %i32_lhs, %i32_rhs : i32, i32
// CHECK: %log_or = emitc.logical_or %i32_lhs, %i32_rhs : i32, i32
// CHECK-GENERIC: %log_or = "emitc.logical_or"(%i32_lhs, %i32_rhs) : (i32, i32)
