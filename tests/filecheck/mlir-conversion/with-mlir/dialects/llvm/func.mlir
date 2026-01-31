// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt | filecheck %s

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
// CHECK-NEXT:    llvm.func @add(%arg0 : i32 {llvm.noundef}, %arg1 : i32 {llvm.noundef}) -> i32 attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, no_inline, no_unwind, optimize_none, passthrough = [["no-trapping-math", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+mmx"]>, tune_cpu = "generic", unnamed_addr = 0 : i64} {
// CHECK-NEXT:      llvm.return %arg0 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
