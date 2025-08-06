// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

// CHECK: target features must start with '+' or '-'
"test.op"() {
    target_features = #llvm.target_features<["-one", "+two", "no-dashes-or-plus"]>,
} : () -> ()
