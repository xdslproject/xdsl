// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

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
