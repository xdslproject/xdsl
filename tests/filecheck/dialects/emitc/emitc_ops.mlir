// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%i32_lhs, %i32_rhs = "test.op"() : () -> (i32, i32)
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
// LiteralOp
//===----------------------------------------------------------------------===//

%lit = emitc.literal "1" : i32
// CHECK: %lit = emitc.literal "1" : i32
// CHECK-GENERIC: %lit = "emitc.literal"() <{value = "1"}> : () -> i32

%lit_pi = emitc.literal "M_PI" : f32
// CHECK: %lit_pi = emitc.literal "M_PI" : f32
// CHECK-GENERIC: %lit_pi = "emitc.literal"() <{value = "M_PI"}> : () -> f32
