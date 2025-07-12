// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

//===----------------------------------------------------------------------===//
// CallOpaqueOp
//===----------------------------------------------------------------------===//

%0 = "emitc.call_opaque"() {callee = "blah"} : () -> i64
emitc.call_opaque "foo" (%0) {args = [
  0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
]} : (i64) -> ()
emitc.call_opaque "test" ()  : () -> ()

// CHECK: builtin.module {
// CHECK-NEXT:  %0 = "emitc.call_opaque"() <{callee = "blah"}> : () -> i64
// CHECK-NEXT:  "emitc.call_opaque"(%0) <{callee = "foo", args = [0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index]}> : (i64) -> ()
// CHECK-NEXT:  "emitc.call_opaque"() <{callee = "test"}> : () -> ()
// CHECK-NEXT: }
