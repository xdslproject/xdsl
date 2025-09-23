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
