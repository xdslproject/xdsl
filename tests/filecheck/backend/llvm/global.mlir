// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  llvm.mlir.global external @x() {addr_space = 0 : i32} : i32
  llvm.mlir.global internal constant @c() {addr_space = 0 : i32} : i64
}

// CHECK: @"x" = external global i32
// CHECK: @"c" = internal constant i64 undef
