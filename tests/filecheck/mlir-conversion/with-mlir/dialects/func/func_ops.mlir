// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | filecheck %s

builtin.module {

  func.func @noarg_void() {
    func.return
  }

   // CHECK:      func.func @noarg_void() {
   // CHECK-NEXT:   func.return
   // CHECK-NEXT: }

  func.func @call_void() {
    func.call @call_void() : () -> ()
    func.return
  }

   // CHECK: func.func @call_void() {
   // CHECK-NEXT:   func.call @call_void() : () -> ()
   // CHECK-NEXT:   func.return
   // CHECK-NEXT: }

  func.func @call_void_attributes() {
    func.call @call_void_attributes() {"hello" = "world"} : () -> ()
    func.return
  }

   // CHECK: func.func @call_void_attributes() {
   // CHECK-NEXT:   func.call @call_void_attributes() {hello = "world"} : () -> ()
   // CHECK-NEXT:   func.return
   // CHECK-NEXT: }

  func.func @arg_rec(%arg0 : i32) -> i32 {
    %1 = func.call @arg_rec(%arg0) : (i32) -> i32
    func.return %1 : i32
  }

   // CHECK: func.func @arg_rec(%{{.*}} : i32) -> i32 {
   // CHECK-NEXT:   %{{.*}} = func.call @arg_rec(%{{.*}}) : (i32) -> i32
   // CHECK-NEXT:   func.return %{{.*}} : i32
   // CHECK-NEXT: }

  func.func @arg_rec_block(i32) -> i32 {
  ^bb0(%2 : i32):
    %3 = "func.call"(%2) {"callee" = @arg_rec_block} : (i32) -> i32
    func.return %3 : i32
  }

  // CHECK: func.func @arg_rec_block(%{{.*}} : i32) -> i32 {
  // CHECK-NEXT:   %{{.*}} = func.call @arg_rec_block(%{{.*}}) : (i32) -> i32
  // CHECK-NEXT:   func.return %{{.*}} : i32
  // CHECK-NEXT: }

  func.func private @external_fn(i32) -> (i32, i32)
  // CHECK: func.func private @external_fn(i32) -> (i32, i32)

  func.func @multi_return_body(%a : i32) -> (i32, i32) {
    func.return %a, %a : i32, i32
  }

  // CHECK: func.func @multi_return_body(%{{.*}} : i32) -> (i32, i32) {
  // CHECK-NEXT:   func.return %{{.*}}, %{{.*}} : i32, i32
  // CHECK-NEXT: }

  func.func public @arg_attrs(%X: tensor<8x8xf64> {"llvm.noalias"},
                              %Y: tensor<8x8xf64> {"llvm.noalias"},
                              %Z: tensor<8x8xf64> {"llvm.noalias"}) -> (tensor<8x8xf64>) {
      func.return %X : tensor<8x8xf64>
  }

  // CHECK:       func.func public @arg_attrs(%{{.*}}: tensor<8x8xf64> {llvm.noalias}, %{{.*}}: tensor<8x8xf64> {llvm.noalias}, %{{.*}}: tensor<8x8xf64> {llvm.noalias}) -> tensor<8x8xf64> {
  // CHECK-NEXT:      func.return %{{.*}} : tensor<8x8xf64>
  // CHECK-NEXT:  }

  func.func @output_attributes() -> (f32 {dialect.a = 0 : i32}, f32 {dialect.b = 0 : i32, dialect.c = 1 : i64}) {
    %r1, %r2 = "test.op"() : () -> (f32, f32)
    return %r1, %r2 : f32, f32
  }

  // CHECK:       func.func @output_attributes() -> (f32 {dialect.a = 0 : i32}, f32 {dialect.b = 0 : i32, dialect.c = 1 : i64}) {
  // CHECK-NEXT:    %0, %1 = "test.op"() : () -> (f32, f32)
  // CHECK-NEXT:    func.return %0, %1 : f32, f32
  // CHECK-NEXT:  }
}
