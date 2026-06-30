// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  llvm.mlir.global external @x() {addr_space = 0 : i32} : i32
  llvm.mlir.global internal constant @c() {addr_space = 0 : i32} : i64
  llvm.mlir.global internal constant @answer(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global private @pi(3.14 : f64) {addr_space = 0 : i32} : f64
  llvm.mlir.global internal constant @hi(dense<[72, 105, 0]> : tensor<3xi8>) {addr_space = 0 : i32} : !llvm.array<3 x i8>
  llvm.mlir.global internal constant @msg("Hi\00") {addr_space = 0 : i32} : !llvm.array<3 x i8>
}

// CHECK: @"x" = external global i32
// CHECK: @"c" = internal constant i64 undef
// CHECK: @"answer" = internal constant i32 42
// CHECK: @"pi" = private global double 0x40091eb851eb851f
// CHECK: @"hi" = internal constant [3 x i8] [i8 72, i8 105, i8 0]
// CHECK: @"msg" = internal constant [3 x i8] c"Hi\00"
