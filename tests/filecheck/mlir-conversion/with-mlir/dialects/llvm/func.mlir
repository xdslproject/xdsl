// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt | filecheck %s

// This test checks that the llvm.func operation can be parse the following
// properties.

"llvm.func"()
    <{
        CConv = #llvm.cconv<ccc>,
        arg_attrs = [{llvm.noundef}, {llvm.noundef}],
        res_attrs = [{llvm.noundef}],
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
// CHECK-SAME:            CConv = #llvm.cconv<ccc>,
// CHECK-SAME:            arg_attrs = [{llvm.noundef}, {llvm.noundef}],
// CHECK-SAME:            frame_pointer = #llvm.framePointerKind<"non-leaf">,
// CHECK-SAME:            function_type = !llvm.func<i32 (i32, i32)>,
// CHECK-SAME:            linkage = #llvm.linkage<"external">,
// CHECK-SAME:            no_inline,
// CHECK-SAME:            no_unwind,
// CHECK-SAME:            optimize_none,
// CHECK-SAME:            passthrough = [["no-trapping-math", "true"]],
// CHECK-SAME:            res_attrs = [{llvm.noundef}],
// CHECK-SAME:            sym_name = "add",
// CHECK-SAME:            target_cpu = "x86-64",
// CHECK-SAME:            target_features = #llvm.target_features<["+mmx"]>,
// CHECK-SAME:            tune_cpu = "generic",
// CHECK-SAME:            unnamed_addr = 0 : i64,
// CHECK-SAME:            visibility_ = 0 : i64
// CHECK-SAME:         }> ({
// CHECK-NEXT:    ^bb0(%arg0 : i32, %arg1 : i32):
// CHECK-NEXT:      llvm.return %arg0 : i32
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }
