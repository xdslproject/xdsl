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

// -----

func.func @gep_indices_type_validation() {
  %ptr = "test.op"() : () -> !llvm.ptr
  // expected-error @+1 {{'rawConstantIndices' expected data of type builtin.i32, but got builtin.i64}}
  %gep = "llvm.getelementptr"(%ptr) <{elem_type = i32, noWrapFlags = 0 : i32, rawConstantIndices = array<i64: 1>}> : (!llvm.ptr) -> !llvm.ptr
}

// -----

func.func @gep_scalar_not_indexable() {
  %ptr = "test.op"() : () -> !llvm.ptr
  %gep = "llvm.getelementptr"(%ptr) <{elem_type = i32, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: 1, 2>}> : (!llvm.ptr) -> !llvm.ptr
  func.return
}

// CHECK: GEP index #1: cannot index into i32

// -----

func.func @gep_struct_ssa_index() {
  %ptr = "test.op"() : () -> !llvm.ptr
  %idx = "test.op"() : () -> i32
  %gep = "llvm.getelementptr"(%ptr, %idx) <{elem_type = !llvm.struct<(i32, i32)>, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: 0, -2147483648>}> : (!llvm.ptr, i32) -> !llvm.ptr
  func.return
}

// CHECK: GEP index #1: struct indices must be constants

// -----

func.func @gep_struct_out_of_range() {
  %ptr = "test.op"() : () -> !llvm.ptr
  %gep = "llvm.getelementptr"(%ptr) <{elem_type = !llvm.struct<(i32, i32)>, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: 0, 5>}> : (!llvm.ptr) -> !llvm.ptr
  func.return
}

// CHECK: GEP index #1: 5 is out of range for struct with 2 field(s)
