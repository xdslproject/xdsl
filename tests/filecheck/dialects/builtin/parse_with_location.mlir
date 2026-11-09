// RUN: xdsl-opt --print-op-generic %s | filecheck %s
// RUN: xdsl-opt --print-op-generic --print-debuginfo %s | xdsl-opt --print-op-generic --print-debuginfo | filecheck %s --check-prefix=CHECK-LOC
//     CHECK: "builtin.module"() ({
// CHECK-LOC: "builtin.module"() ({
"builtin.module"() ({
  //     CHECK: ^{{.*}}(%{{.*}}: i32):
  // CHECK-LOC: ^{{.*}}(%{{.*}}: i32 loc(unknown)):
  ^bb0(%arg: i32 loc(unknown)):
    //     CHECK: "test.op"() : () -> !test.type<"int">
    // CHECK-LOC: "test.op"() : () -> !test.type<"int"> loc(unknown)
    %0 = "test.op"() : () -> !test.type<"int"> loc(unknown)
    //     CHECK: "test.op"() : () -> ()
    // CHECK-LOC: "test.op"() : () -> () loc(unknown)
    "test.op"() : () -> () loc(unknown)

    //     CHECK: "test.op"() : () -> ()
    // CHECK-LOC: "test.op"() : () -> () loc(fused<"metadata">[unknown])
    "test.op"() : () -> () loc(fused<"metadata">[unknown])

    //     CHECK: "test.op"() : () -> ()
    // CHECK-LOC: "test.op"() : () -> () loc(fused<{asm.comment = ""}>[unknown])
    "test.op"() : () -> () loc(fused<{asm.comment = ""}>[unknown])

    //     CHECK: "test.op"() : () -> ()
    // CHECK-LOC: "test.op"() : () -> () loc(fused<{asm.comment = "assembly comment", tag = "keep"}>["source":2:3])
    "test.op"() : () -> () loc(fused<{asm.comment = "assembly comment", tag = "keep"}>["source":2:3])

    //     CHECK: "test.op"() : () -> ()
    // CHECK-LOC: "test.op"() : () -> () loc(fused<{asm.comment = "assembly comment"}>[fused<"existing metadata">["source":2:3]])
    "test.op"() : () -> () loc(fused<{asm.comment = "assembly comment"}>[fused<"existing metadata">["source":2:3]])
//     CHECK: }) : () -> ()
// CHECK-LOC: }) : () -> () loc(unknown)
}) : () -> ()
