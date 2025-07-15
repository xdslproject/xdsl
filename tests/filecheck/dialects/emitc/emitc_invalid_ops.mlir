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
