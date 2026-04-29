// RUN: XDSL_ROUNDTRIP
builtin.module {
  llvm.mlir.global internal constant @str0("Hello world!\n") {addr_space = 0 : i32} : !llvm.array<13 x i8>
  llvm.mlir.global external @x() {addr_space = 0 : i32} : i32
  llvm.mlir.global private @y(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external thread_local @tl() {addr_space = 0 : i32} : i32
  llvm.mlir.global external unnamed_addr @u() {addr_space = 0 : i32} : i32
  llvm.mlir.global external local_unnamed_addr @l() {addr_space = 0 : i32} : i32
  llvm.mlir.global external @g_with_init() {addr_space = 0 : i32} : i64 {
    %0 = llvm.mlir.constant(42 : i64) : i64
    llvm.return %0 : i64
  }
  llvm.mlir.global internal constant @str_inferred("test") {addr_space = 0 : i32}
  llvm.mlir.global private @val_inferred(42 : i32) {addr_space = 0 : i32}
  %0 = llvm.mlir.addressof @str0 : !llvm.ptr
  %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<13 x i8>
}

// CHECK: builtin.module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("Hello world!\n") {addr_space = 0 : i32} : !llvm.array<13 x i8>
// CHECK-NEXT:   llvm.mlir.global external @x() {addr_space = 0 : i32} : i32
// CHECK-NEXT:   llvm.mlir.global private @y(42 : i32) {addr_space = 0 : i32} : i32
// CHECK-NEXT:   llvm.mlir.global external thread_local @tl() {addr_space = 0 : i32} : i32
// CHECK-NEXT:   llvm.mlir.global external unnamed_addr @u() {addr_space = 0 : i32} : i32
// CHECK-NEXT:   llvm.mlir.global external local_unnamed_addr @l() {addr_space = 0 : i32} : i32
// CHECK-NEXT:   llvm.mlir.global external @g_with_init() {addr_space = 0 : i32} : i64 {
// CHECK-NEXT:     %0 = llvm.mlir.constant(42) : i64
// CHECK-NEXT:     llvm.return %0 : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.mlir.global internal constant @str_inferred("test") {addr_space = 0 : i32} : !llvm.array<4 x i8>
// CHECK-NEXT:   llvm.mlir.global private @val_inferred(42 : i32) {addr_space = 0 : i32} : i32
// CHECK-NEXT:   %1 = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK-NEXT:   %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<13 x i8>
// CHECK-NEXT: }
