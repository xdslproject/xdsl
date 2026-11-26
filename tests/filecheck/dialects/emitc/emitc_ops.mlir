// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%i32_lhs, %i32_rhs = "test.op"() : () -> (i32, i32)
%f32_lhs, %f32_rhs = "test.op"() : () -> (f32, f32)
%size_lhs, %size_rhs = "test.op"() : () -> (!emitc.size_t, !emitc.size_t)
%ptr_f32, %i32_offset = "test.op"() : () -> (!emitc.ptr<f32>, i32)
%opaque_uint = "test.op"() : () -> !emitc.opaque<"unsigned int">
%tensor_lhs, %tensor_rhs = "test.op"() : () -> (tensor<3x4xi32>, tensor<3x4xi32>)
%lvalue_i32 = "test.op"() : () -> !emitc.lvalue<i32>
%lvalue_ptr = "test.op"() : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>

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
// AddressOfOp
//===----------------------------------------------------------------------===//

%address_of = emitc.address_of %lvalue_i32 : !emitc.lvalue<i32>
// CHECK: %address_of = emitc.address_of %lvalue_i32 : !emitc.lvalue<i32>
// CHECK-GENERIC: %address_of = "emitc.address_of"(%lvalue_i32) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

%address_of_ptr = emitc.address_of %lvalue_ptr : !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>
// CHECK: %address_of_ptr = emitc.address_of %lvalue_ptr : !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>
// CHECK-GENERIC: %address_of_ptr = "emitc.address_of"(%lvalue_ptr) : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>) -> !emitc.ptr<!emitc.ptr<!emitc.opaque<"int32_t">>>

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

%assign_var, %assign_value = "test.op"() : () -> (!emitc.lvalue<i32>, i32)

emitc.assign %assign_value : i32 to %assign_var : !emitc.lvalue<i32>
// CHECK: emitc.assign %assign_value : i32 to %assign_var : <i32>
// CHECK-GENERIC: "emitc.assign"(%assign_var, %assign_value) : (!emitc.lvalue<i32>, i32) -> ()

//===----------------------------------------------------------------------===//
// BitwiseLeftShiftOp
//===----------------------------------------------------------------------===//

%bitwise_left_shift = emitc.bitwise_left_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %bitwise_left_shift = emitc.bitwise_left_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %bitwise_left_shift = "emitc.bitwise_left_shift"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseNotOp
//===----------------------------------------------------------------------===//

%bitwise_not = emitc.bitwise_not %i32_lhs : (i32) -> i32
// CHECK: %bitwise_not = emitc.bitwise_not %i32_lhs : (i32) -> i32
// CHECK-GENERIC: %bitwise_not = "emitc.bitwise_not"(%i32_lhs) : (i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseOrOp
//===----------------------------------------------------------------------===//

%bitwise_or = emitc.bitwise_or %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %bitwise_or = emitc.bitwise_or %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %bitwise_or = "emitc.bitwise_or"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseRightShiftOp
//===----------------------------------------------------------------------===//

%bitwise_right_shift = emitc.bitwise_right_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %bitwise_right_shift = emitc.bitwise_right_shift %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %bitwise_right_shift = "emitc.bitwise_right_shift"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// BitwiseXorOp
//===----------------------------------------------------------------------===//

%bitwise_xor = emitc.bitwise_xor %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %bitwise_xor = emitc.bitwise_xor %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %bitwise_xor = "emitc.bitwise_xor"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

%cons_int = "emitc.constant"() <{value = 42 : i32}> : () -> i32
// CHECK: %cons_int = "emitc.constant"() <{value = 42 : i32}> : () -> i32
// CHECK-GENERIC: %cons_int = "emitc.constant"() <{value = 42 : i32}> : () -> i32

%cons_zero = "emitc.constant"() <{value = 0 : i32}> : () -> i32
// CHECK: %cons_zero = "emitc.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-GENERIC: %cons_zero = "emitc.constant"() <{value = 0 : i32}> : () -> i32

%cons_float = "emitc.constant"() <{value = 1.500000e+00 : f32}> : () -> f32
// CHECK: %cons_float = "emitc.constant"() <{value = 1.500000e+00 : f32}> : () -> f32
// CHECK-GENERIC: %cons_float = "emitc.constant"() <{value = 1.500000e+00 : f32}> : () -> f32

%cons_sizet = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.size_t
// CHECK: %cons_sizet = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.size_t
// CHECK-GENERIC: %cons_sizet = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.size_t

%cons_ssizet = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ssize_t
// CHECK: %cons_ssizet = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ssize_t
// CHECK-GENERIC: %cons_ssizet = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ssize_t

%cons_ptrdiff = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ptrdiff_t
// CHECK: %cons_ptrdiff = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ptrdiff_t
// CHECK-GENERIC: %cons_ptrdiff = "emitc.constant"() <{value = 42 : index}> : () -> !emitc.ptrdiff_t

%cons_opaque = "emitc.constant"() <{value = #emitc.opaque<"42">}> : () -> !emitc.opaque<"int">
// CHECK: %cons_opaque = "emitc.constant"() <{value = #emitc.opaque<"42">}> : () -> !emitc.opaque<"int">
// CHECK-GENERIC: %cons_opaque = "emitc.constant"() <{value = #emitc.opaque<"42">}> : () -> !emitc.opaque<"int">

//===----------------------------------------------------------------------===//
// DereferenceOp
//===----------------------------------------------------------------------===//

%dereference = emitc.dereference %ptr_f32 : !emitc.ptr<f32>
// CHECK: %dereference = emitc.dereference %ptr_f32 : !emitc.ptr<f32>
// CHECK-GENERIC: %dereference = "emitc.dereference"(%ptr_f32) : (!emitc.ptr<f32>) -> !emitc.lvalue<f32>

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

%div_int = emitc.div %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %div_int = emitc.div %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %div_int = "emitc.div"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

%div_float = emitc.div %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK: %div_float = emitc.div %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK-GENERIC: %div_float = "emitc.div"(%f32_lhs, %f32_rhs) : (f32, f32) -> f32

%div_size = emitc.div %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK: %div_size = emitc.div %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK-GENERIC: %div_size = "emitc.div"(%size_lhs, %size_rhs) : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t

//===----------------------------------------------------------------------===//
// LiteralOp
//===----------------------------------------------------------------------===//

%lit = emitc.literal "1" : i32
// CHECK: %lit = emitc.literal "1" : i32
// CHECK-GENERIC: %lit = "emitc.literal"() <{value = "1"}> : () -> i32

%lit_pi = emitc.literal "M_PI" : f32
// CHECK: %lit_pi = emitc.literal "M_PI" : f32
// CHECK-GENERIC: %lit_pi = "emitc.literal"() <{value = "M_PI"}> : () -> f32

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

%load = emitc.load %lvalue_i32 : !emitc.lvalue<i32>
// CHECK: %load = emitc.load %lvalue_i32 : <i32>
// CHECK-GENERIC: %load = "emitc.load"(%lvalue_i32) : (!emitc.lvalue<i32>) -> i32

%load_ptr = emitc.load %lvalue_ptr : !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>
// CHECK: %load_ptr = emitc.load %lvalue_ptr : <!emitc.ptr<!emitc.opaque<"int32_t">>>
// CHECK-GENERIC: %load_ptr = "emitc.load"(%lvalue_ptr) : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>) -> !emitc.ptr<!emitc.opaque<"int32_t">>

//===----------------------------------------------------------------------===//
// LogicalAndOp
//===----------------------------------------------------------------------===//

%logical_and = emitc.logical_and %i32_lhs, %i32_rhs : i32, i32
// CHECK: %logical_and = emitc.logical_and %i32_lhs, %i32_rhs : i32, i32
// CHECK-GENERIC: %logical_and = "emitc.logical_and"(%i32_lhs, %i32_rhs) : (i32, i32) -> i1

//===----------------------------------------------------------------------===//
// LogicalNotOp
//===----------------------------------------------------------------------===//

%logical_not = emitc.logical_not %i32_lhs : i32
// CHECK: %logical_not = emitc.logical_not %i32_lhs : i32
// CHECK-GENERIC: %logical_not = "emitc.logical_not"(%i32_lhs) : (i32) -> i1

//===----------------------------------------------------------------------===//
// LogicalOrOp
//===----------------------------------------------------------------------===//

%logical_or = emitc.logical_or %i32_lhs, %i32_rhs : i32, i32
// CHECK: %logical_or = emitc.logical_or %i32_lhs, %i32_rhs : i32, i32
// CHECK-GENERIC: %logical_or = "emitc.logical_or"(%i32_lhs, %i32_rhs) : (i32, i32) -> i1

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

%mul_int = emitc.mul %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %mul_int = emitc.mul %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %mul_int = "emitc.mul"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

%mul_float = emitc.mul %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK: %mul_float = emitc.mul %f32_lhs, %f32_rhs : (f32, f32) -> f32
// CHECK-GENERIC: %mul_float = "emitc.mul"(%f32_lhs, %f32_rhs) : (f32, f32) -> f32

%mul_size = emitc.mul %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK: %mul_size = emitc.mul %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK-GENERIC: %mul_size = "emitc.mul"(%size_lhs, %size_rhs) : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t

//===----------------------------------------------------------------------===//
// RemOp
//===----------------------------------------------------------------------===//

%rem_int = emitc.rem %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK: %rem_int = emitc.rem %i32_lhs, %i32_rhs : (i32, i32) -> i32
// CHECK-GENERIC: %rem_int = "emitc.rem"(%i32_lhs, %i32_rhs) : (i32, i32) -> i32

%rem_size = emitc.rem %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK: %rem_size = emitc.rem %size_lhs, %size_rhs : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
// CHECK-GENERIC: %rem_size = "emitc.rem"(%size_lhs, %size_rhs) : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t

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

%sub_ptr_ptr = emitc.sub %ptr_f32, %ptr_f32 : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptrdiff_t
// CHECK: %sub_ptr_ptr = emitc.sub %ptr_f32, %ptr_f32 : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptrdiff_t
// CHECK-GENERIC: %sub_ptr_ptr = "emitc.sub"(%ptr_f32, %ptr_f32) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptrdiff_t

%sub_tensor = emitc.sub %tensor_lhs, %tensor_rhs : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
// CHECK: %sub_tensor = emitc.sub %tensor_lhs, %tensor_rhs : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
// CHECK-GENERIC: %sub_tensor = "emitc.sub"(%tensor_lhs, %tensor_rhs) : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>

//===----------------------------------------------------------------------===//
// UnaryMinusOp
//===----------------------------------------------------------------------===//

%unary_minus = emitc.unary_minus %i32_lhs : (i32) -> i32
// CHECK: %unary_minus = emitc.unary_minus %i32_lhs : (i32) -> i32
// CHECK-GENERIC: %unary_minus = "emitc.unary_minus"(%i32_lhs) : (i32) -> i32

//===----------------------------------------------------------------------===//
// UnaryPlusOp
//===----------------------------------------------------------------------===//

%unary_plus = emitc.unary_plus %i32_lhs : (i32) -> i32
// CHECK: %unary_plus = emitc.unary_plus %i32_lhs : (i32) -> i32
// CHECK-GENERIC: %unary_plus = "emitc.unary_plus"(%i32_lhs) : (i32) -> i32

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

%variable = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK: %variable = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK-GENERIC: %variable = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>

%variable_init = "emitc.variable"() <{value = 7 : i32}> : () -> !emitc.lvalue<i32>
// CHECK: %variable_init = "emitc.variable"() <{value = 7 : i32}> : () -> !emitc.lvalue<i32>
// CHECK-GENERIC: %variable_init = "emitc.variable"() <{value = 7 : i32}> : () -> !emitc.lvalue<i32>

%variable_zero = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
// CHECK: %variable_zero = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
// CHECK-GENERIC: %variable_zero = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>

%variable_sizet = "emitc.variable"() <{value = 42 : index}> : () -> !emitc.lvalue<!emitc.size_t>
// CHECK: %variable_sizet = "emitc.variable"() <{value = 42 : index}> : () -> !emitc.lvalue<!emitc.size_t>
// CHECK-GENERIC: %variable_sizet = "emitc.variable"() <{value = 42 : index}> : () -> !emitc.lvalue<!emitc.size_t>

%variable_ptr = "emitc.variable"() <{value = #emitc.opaque<"NULL">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>
// CHECK: %variable_ptr = "emitc.variable"() <{value = #emitc.opaque<"NULL">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>
// CHECK-GENERIC: %variable_ptr = "emitc.variable"() <{value = #emitc.opaque<"NULL">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>

%variable_array = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<2xi32>
// CHECK: %variable_array = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<2xi32>
// CHECK-GENERIC: %variable_array = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<2xi32>
