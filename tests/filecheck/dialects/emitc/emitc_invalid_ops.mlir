// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

//===----------------------------------------------------------------------===//
// CallOpaqueOp
//===----------------------------------------------------------------------===//

// CHECK: index argument is out of range
emitc.call_opaque "test" () {args = [0 : index]} : () -> ()

// -----

// CHECK: index argument is out of range
%arg = "test.op"() : () -> i32
emitc.call_opaque "test" (%arg, %arg) {args = [2 : index]} : (i32, i32) -> ()

// -----

// CHECK: callee must not be empty
emitc.call_opaque "" () : () -> ()

// -----

%arg = "test.op"() : () -> i32
// CHECK: array argument has no type
emitc.call_opaque "nonetype_arg"(%arg) {args = [0 : index, [0, 1, 2]]} : (i32) -> i32

// -----

%arg = "test.op"() : () -> i32
// CHECK: template argument has invalid type
emitc.call_opaque "nonetype_template_arg"(%arg) {template_args = [[0, 1, 2]]} : (i32) -> i32

// -----

%arg = "test.op"() : () -> i32
// CHECK: template argument has invalid type
emitc.call_opaque "dense_template_argument"(%arg) {template_args = [dense<[1.0, 1.0]> : tensor<2xf32>]} : (i32) -> i32

// -----

// CHECK: cannot return array type
emitc.call_opaque "array_result"() : () -> !emitc.array<4xi32>

// -----

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

%ptr_lhs, %ptr_rhs = "test.op"() : () -> (!emitc.ptr<f32>, !emitc.ptr<f32>)
// CHECK: emitc.add requires that at most one operand is a pointer
%two_ptrs = emitc.add %ptr_lhs, %ptr_rhs : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>

// -----

%ptr, %float_val = "test.op"() : () -> (!emitc.ptr<f32>, f32)
// CHECK: emitc.add requires that one operand is an integer or of opaque type if the other is a pointer
%ptr_float = emitc.add %ptr, %float_val : (!emitc.ptr<f32>, f32) -> !emitc.ptr<f32>

// -----

%float_val, %ptr = "test.op"() : () -> (f32, !emitc.ptr<f32>)
// CHECK: emitc.add requires that one operand is an integer or of opaque type if the other is a pointer
%float_ptr = emitc.add %float_val, %ptr : (f32, !emitc.ptr<f32>) -> !emitc.ptr<f32>

// -----

%dynamic_tensor_lhs, %dynamic_tensor_rhs = "test.op"() : () -> (tensor<?x4xi32>, tensor<?x4xi32>)
// CHECK: Type tensor<?x4xi32> is not a supported EmitC type
%add_dynamic_tensor = emitc.add %dynamic_tensor_lhs, %dynamic_tensor_rhs : (tensor<?x4xi32>, tensor<?x4xi32>) -> tensor<?x4xi32>

// -----

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

%lvalue = "test.op"() : () -> !emitc.lvalue<i32>
// CHECK: applicable operator must not be empty
%empty_op = emitc.apply ""(%lvalue) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

// -----

%lvalue = "test.op"() : () -> !emitc.lvalue<i32>
// CHECK: applicable operator is illegal
%illegal_op = emitc.apply "+"(%lvalue) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

// -----

%not_lvalue = "test.op"() : () -> i32
// CHECK: operand type must be an lvalue when applying `&`
%wrong_type_addressof = emitc.apply "&"(%not_lvalue) : (i32) -> !emitc.ptr<i32>

// -----

%lvalue = "test.op"() : () -> !emitc.lvalue<i32>
// CHECK: result type must be a pointer when applying `&`
%wrong_result_addressof = emitc.apply "&"(%lvalue) : (!emitc.lvalue<i32>) -> i32

// -----

%not_ptr = "test.op"() : () -> i32
// CHECK: operand type must be a pointer when applying `*`
%wrong_type_deref = emitc.apply "*"(%not_ptr) : (i32) -> i32

// -----

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// CHECK: requires attribute to either be an #emitc.opaque attribute or it's type (i64) to match the op's result type (i32)
%c = "emitc.constant"() <{value = 42 : i64}> : () -> i32

// -----

// CHECK: requires attribute to either be an #emitc.opaque attribute or it's type (i32) to match the op's result type (!emitc.size_t)
%c = "emitc.constant"() <{value = 42 : i32}> : () -> !emitc.size_t

// -----

// CHECK: value must not be empty
%c = "emitc.constant"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"int">

// -----

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

// CHECK: requires attribute to either be an #emitc.opaque attribute or it's type (i64) to match the op's result type (i32)
%v = "emitc.variable"() <{value = 42 : i64}> : () -> !emitc.lvalue<i32>

// -----

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

%var = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
%val = "test.op"() : () -> i64
// CHECK: requires value's type (i64) to match variable's type (i32)
emitc.assign %val : i64 to %var : !emitc.lvalue<i32>

// -----

"test.op"() ({
^bb0(%arg: !emitc.lvalue<i32>):
  %val = "test.op"() : () -> i32
  // CHECK: cannot assign to block argument
  emitc.assign %val : i32 to %arg : !emitc.lvalue<i32>
  "test.termop"() : () -> ()
}) : () -> ()

// -----

//===----------------------------------------------------------------------===//
// LiteralOp
//===----------------------------------------------------------------------===//

// CHECK: value must not be empty
%lit = emitc.literal "" : i32

// -----

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

%var = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK: i32 should be of base attribute emitc.ptr
%addr = "emitc.address_of"(%var) : (!emitc.lvalue<i32>) -> i32

// -----

%var = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK: attribute i32 expected from variable 'T', but got f32
%addr = "emitc.address_of"(%var) : (!emitc.lvalue<i32>) -> !emitc.ptr<f32>

// -----

//===----------------------------------------------------------------------===//
// DereferenceOp
//===----------------------------------------------------------------------===//

%not_ptr = "test.op"() : () -> i32
// CHECK: i32 should be of base attribute emitc.ptr
%deref = "emitc.dereference"(%not_ptr) : (i32) -> !emitc.lvalue<i32>

// -----

%ptr = "test.op"() : () -> !emitc.ptr<i32>
// CHECK: attribute i32 expected from variable 'T', but got f32
%deref = "emitc.dereference"(%ptr) : (!emitc.ptr<i32>) -> !emitc.lvalue<f32>

// -----

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

%var = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK: attribute i32 expected from variable 'T', but got f32
%loaded = "emitc.load"(%var) : (!emitc.lvalue<i32>) -> f32
