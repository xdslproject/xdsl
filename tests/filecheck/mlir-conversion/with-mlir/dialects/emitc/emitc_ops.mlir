// RUN: xdsl-opt %s | mlir-opt --allow-unregistered-dialect | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s --print-op-generic | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic | xdsl-opt | filecheck %s

%i32_lhs, %i32_rhs = "test.op"() : () -> (i32, i32)
%ptr_f32, %i32_offset = "test.op"() : () -> (!emitc.ptr<f32>, i32)
%opaque_uint = "test.op"() : () -> !emitc.opaque<"unsigned int">

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
