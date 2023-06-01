// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s

builtin.module {

  func.func @noarg_void() {
    "func.return"() : () -> ()
  }

   // CHECK:      func.func @noarg_void() {
   // CHECK-NEXT:   "func.return"() : () -> ()
   // CHECK-NEXT: }

  func.func @call_void() {
    "func.call"() {"callee" = @call_void} : () -> ()
    "func.return"() : () -> ()
  }

   // CHECK: func.func @call_void() {
   // CHECK-NEXT:   "func.call"() {"callee" = @call_void} : () -> ()
   // CHECK-NEXT:   "func.return"() : () -> ()
   // CHECK-NEXT: }

  func.func @arg_rec(%arg0 : i32) -> i32 {
    %1 = "func.call"(%arg0) {"callee" = @arg_rec} : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }

   // CHECK: func.func @arg_rec(%{{.*}} : i32) -> i32 {
   // CHECK-NEXT:   %{{.*}} = "func.call"(%{{.*}}) {"callee" = @arg_rec} : (i32) -> i32
   // CHECK-NEXT:   "func.return"(%{{.*}}) : (i32) -> ()
   // CHECK-NEXT: }

  func.func @arg_rec_block(i32) -> i32 {
  ^0(%2 : i32):
    %3 = "func.call"(%2) {"callee" = @arg_rec_block} : (i32) -> i32
    "func.return"(%3) : (i32) -> ()
  }

  // CHECK: func.func @arg_rec_block(%{{.*}} : i32) -> i32 {
  // CHECK-NEXT:   %{{.*}} = "func.call"(%{{.*}}) {"callee" = @arg_rec_block} : (i32) -> i32
  // CHECK-NEXT:   "func.return"(%{{.*}}) : (i32) -> ()
  // CHECK-NEXT: }

  func.func private @external_fn(i32) -> (i32, i32)
  // CHECK: func.func private @external_fn(i32) -> (i32, i32)

  func.func @multi_return_body(%a : i32) -> (i32, i32) {
    "func.return"(%a, %a) : (i32, i32) -> ()
  }

  // CHECK: func.func @multi_return_body(%{{.*}} : i32) -> (i32, i32) {
  // CHECK-NEXT:   "func.return"(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
  // CHECK-NEXT: }

}
