// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
    %f = "test.op"() : () -> !llvm.func<i32 (i32, ..., i32)>
}

// CHECK: Varargs specifier `...` must be at the end of the argument definition

// -----

builtin.module {
    %cc = "test.op"() {"cconv" = #llvm.cconv<invalid>} : () -> ()
}

// CHECK: Unknown calling convention

// -----

func.func public @main() {
  %0 = "test.op"() : () -> (!llvm.struct<(i32)>)
  %1 = "llvm.extractvalue"(%0) {"position" = array<i32: 0>} : (!llvm.struct<(i32)>) -> i32
  func.return
}

// CHECK: Expected attribute i64 but got i32

// -----

func.func public @main() {
  %0, %1 = "test.op"() : () -> (!llvm.struct<(i32)>, i32)
  %2 = "llvm.insertvalue"(%0, %1) {"position" = array<i32: 0>} : (!llvm.struct<(i32)>, i32) -> !llvm.struct<(i32)>
  func.return
}

// CHECK: Expected attribute i64 but got i32
