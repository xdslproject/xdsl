// RUN: xdsl-opt --print-op-generic %s --print-debuginfo | $XDSL_MLIR_OPT --mlir-print-op-generic --allow-unregistered-dialect --mlir-print-debuginfo --mlir-print-local-scope | xdsl-opt --print-op-generic --print-debuginfo | filecheck %s

"test.op"() : () -> () loc(unknown)

// CHECK: "test.op"() : () -> () loc(unknown)

"test.op"() : () -> () loc(fused<"metadata">["source":2:3])
// CHECK: "test.op"() : () -> () loc(fused<"metadata">["source":2:3])

"test.op"() : () -> () loc(fused<{tag = "example"}>["source":2:3])
// CHECK: "test.op"() : () -> () loc(fused<{tag = "example"}>["source":2:3])

"test.op"() : () -> () loc(fused<{asm.comment = ""}>[unknown])
// CHECK: "test.op"() : () -> () loc(fused<{asm.comment = ""}>[unknown])

"test.op"() : () -> () loc(fused<{asm.comment = "assembly comment"}>[unknown])
// CHECK: "test.op"() : () -> () loc(fused<{asm.comment = "assembly comment"}>[unknown])

"test.op"() : () -> () loc(fused<{asm.comment = "assembly comment", tag = "keep"}>["source":2:3, "other":7:5])
// CHECK: "test.op"() : () -> () loc(fused<{asm.comment = "assembly comment", tag = "keep"}>["source":2:3, "other":7:5])

"test.op"() : () -> () loc(fused<{asm.comment = "assembly comment"}>[fused<"existing metadata">["source":2:3]])
// CHECK: "test.op"() : () -> () loc(fused<{asm.comment = "assembly comment"}>[fused<"existing metadata">["source":2:3]])
