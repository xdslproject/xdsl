// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP
builtin.module {
  llvm.mlir.global internal constant @str0("Hello world!") {addr_space = 0 : i32} : !llvm.array<12 x i8>
  llvm.mlir.global internal constant @data(0 : i32) {addr_space = 0 : i32} : i32
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
}

// CHECK:      builtin.module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("Hello world!") {{.*}}: !llvm.array<12 x i8>
// CHECK-NEXT:   llvm.mlir.global internal constant @data(0 : i32) {{.*}}: i32
// CHECK-NEXT:   llvm.mlir.global external @x() {{.*}}: i32
// CHECK-NEXT:   llvm.mlir.global private @y(42 : i32) {{.*}}: i32
// CHECK-NEXT:   llvm.mlir.global external thread_local @tl() {{.*}}: i32
// CHECK-NEXT:   llvm.mlir.global external unnamed_addr @u() {{.*}}: i32
// CHECK-NEXT:   llvm.mlir.global external local_unnamed_addr @l() {{.*}}: i32
// CHECK-NEXT:   llvm.mlir.global external @g_with_init() {{.*}}: i64 {
// CHECK-NEXT:     %{{.*}} = llvm.mlir.constant(42{{.*}}) : i64
// CHECK-NEXT:     llvm.return %{{.*}} : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.mlir.global internal constant @str_inferred("test") {{.*}}
// CHECK-NEXT:   llvm.mlir.global private @val_inferred(42 : i32) {{.*}}
// CHECK-NEXT: }
