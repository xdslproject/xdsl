// RUN: MLIR_GENERIC_ROUNDTRIP

// CHECK: frame_all = #llvm.framePointerKind<all>
// CHECK: frame_non_leaf = #llvm.framePointerKind<"non-leaf">
// CHECK: frame_none = #llvm.framePointerKind<none>
// CHECK: frame_reserved = #llvm.framePointerKind<reserved>
"test.op"() {
    frame_all = #llvm.framePointerKind<all>,
    frame_non_leaf = #llvm.framePointerKind<"non-leaf">,
    frame_none = #llvm.framePointerKind<none>,
    frame_reserved = #llvm.framePointerKind<reserved>
    }: () -> ()

// CHECK: target_features = #llvm.target_features<["-one", "+two"]>
// CHECK: target_features_empty = #llvm.target_features<[]>
"test.op"() {
    target_features = #llvm.target_features<["-one", "+two"]>,
    target_features_empty = #llvm.target_features<[]>
} : () -> ()
