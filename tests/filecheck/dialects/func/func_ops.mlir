// RUN: XDSL_ROUNDTRIP

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

  func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
    %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
    func.return %1 : !test.type<"int">
  }

   // CHECK: func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
   // CHECK-NEXT:   %{{.*}} = func.call @arg_rec(%{{.*}}) : (!test.type<"int">) -> !test.type<"int">
   // CHECK-NEXT:   func.return %{{.*}} : !test.type<"int">
   // CHECK-NEXT: }

  func.func @arg_rec_block(!test.type<"int">) -> !test.type<"int"> {
  ^bb0(%2 : !test.type<"int">):
    %3 = func.call @arg_rec_block(%2) : (!test.type<"int">) -> !test.type<"int">
    func.return %3 : !test.type<"int">
  }

  // CHECK: func.func @arg_rec_block(%0 : !test.type<"int">) -> !test.type<"int"> {
  // CHECK-NEXT:   %1 = func.call @arg_rec_block(%0) : (!test.type<"int">) -> !test.type<"int">
  // CHECK-NEXT:   func.return %1 : !test.type<"int">
  // CHECK-NEXT: }

  func.func private @external_fn(i32) -> (i32, i32)
  // CHECK: func.func private @external_fn(i32) -> (i32, i32)

  func.func @multi_return_body(%a : i32) -> (i32, i32) {
    func.return %a, %a : i32, i32
  }

  // CHECK: func.func @multi_return_body(%a : i32) -> (i32, i32) {
  // CHECK-NEXT:   func.return %a, %a : i32, i32
  // CHECK-NEXT: }

  func.func public @arg_attrs(%X: tensor<8x8xf64> {"llvm.noalias"},
                              %Y: tensor<8x8xf64> {"llvm.noalias"},
                              %Z: tensor<8x8xf64> {"llvm.noalias"}) -> tensor<8x8xf64> {
      return %X : tensor<8x8xf64>
  }

  // CHECK:       func.func public @arg_attrs(%{{.*}} : tensor<8x8xf64> {llvm.noalias}, %{{.*}} : tensor<8x8xf64> {llvm.noalias}, %{{.*}} : tensor<8x8xf64> {llvm.noalias}) -> tensor<8x8xf64> {
  // CHECK-NEXT:      return %{{.*}} : tensor<8x8xf64>
  // CHECK-NEXT:  }

  func.func @output_attributes() -> (f32 {dialect.a = 0 : i32}, f32 {dialect.b = 0 : i32, dialect.c = 1 : i64}) {
    %r1, %r2 = "test.op"() : () -> (f32, f32)
    return %r1, %r2 : f32, f32
  }

  // CHECK:       func.func @output_attributes() -> (f32 {dialect.a = 0 : i32}, f32 {dialect.b = 0 : i32, dialect.c = 1 : i64}) {
  // CHECK-NEXT:    %r1, %r2 = "test.op"() : () -> (f32, f32)
  // CHECK-NEXT:    func.return %r1, %r2 : f32, f32
  // CHECK-NEXT:  }

  func.func @output_attribute_single() -> (f32 {dialect.a = 0 : i32}) {
    %r1 = "test.op"() : () -> (f32)
    return %r1: f32
  }

  // CHECK:       func.func @output_attribute_single() -> (f32 {dialect.a = 0 : i32}) {
  // CHECK-NEXT:    %r1 = "test.op"() : () -> f32
  // CHECK-NEXT:    func.return %r1 : f32
  // CHECK-NEXT:  }
}
