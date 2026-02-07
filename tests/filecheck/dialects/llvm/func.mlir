// RUN: xdsl-opt %s | filecheck %s

// This test checks that the llvm.func operation can be parse the following
// properties.

"llvm.func"()
    <{
        CConv = #llvm.cconv<ccc>,
        arg_attrs = [{llvm.noundef}, {llvm.noundef}],
        frame_pointer = #llvm.framePointerKind<"non-leaf">,
        function_type = !llvm.func<i32 (i32, i32)>,
        linkage = #llvm.linkage<external>,
        no_inline,
        no_unwind,
        optimize_none,
        passthrough = [["no-trapping-math", "true"]],
        sym_name = "add",
        target_cpu = "x86-64",
        target_features = #llvm.target_features<["+mmx"]>,
        tune_cpu = "generic",
        unnamed_addr = 0 : i64,
        visibility_ = 0 : i64
    }> ({
  ^bb0(%arg0: i32, %arg1: i32):
    llvm.return %arg0 : i32
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    "llvm.func"() <{
// CHECK:            CConv = #llvm.cconv<ccc>,
// CHECK:            arg_attrs = [{llvm.noundef}, {llvm.noundef}],
// CHECK:            frame_pointer = #llvm.framePointerKind<"non-leaf">,
// CHECK:            function_type = !llvm.func<i32 (i32, i32)>,
// CHECK:            linkage = #llvm.linkage<"external">,
// CHECK:            no_inline,
// CHECK:            no_unwind,
// CHECK:            optimize_none,
// CHECK:            passthrough = [["no-trapping-math", "true"]],
// CHECK:            sym_name = "add",
// CHECK:            target_cpu = "x86-64",
// CHECK:            target_features = #llvm.target_features<["+mmx"]>,
// CHECK:            tune_cpu = "generic",
// CHECK:            unnamed_addr = 0 : i64,
// CHECK:            visibility_ = 0 : i64
// CHECK:         }> ({
// CHECK-NEXT:    ^bb0(%arg0 : i32, %arg1 : i32):
// CHECK-NEXT:      llvm.return %arg0 : i32
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }
