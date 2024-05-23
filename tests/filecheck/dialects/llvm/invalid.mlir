// RUN: xdsl-opt "%s" --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck "%s"

builtin.module {
    %f = "test.op"() : () -> !llvm.func<i32 (i32, ..., i32)>
}

// CHECK: Varargs specifier `...` must be at the end of the argument definition

// -----
// CHECK: -----

builtin.module {
    %cc = "test.op"() {"cconv" = #llvm.cconv<invalid>} : () -> ()
}

// CHECK: Unknown calling convention
