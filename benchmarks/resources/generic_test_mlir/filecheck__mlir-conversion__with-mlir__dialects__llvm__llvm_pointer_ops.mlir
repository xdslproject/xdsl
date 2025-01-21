"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
  %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr
  %2 = "llvm.load"(%1) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
  %3 = "llvm.mlir.null"() : () -> !llvm.ptr
  %4 = "llvm.alloca"(%0) <{alignment = 32 : i64, elem_type = i64}> : (i64) -> !llvm.ptr
  %5 = "llvm.load"(%4) <{ordering = 0 : i64}> : (!llvm.ptr) -> i64
  %6 = "llvm.alloca"(%0) <{alignment = 32 : i64, elem_type = i32}> : (i64) -> !llvm.ptr
  %7 = "llvm.getelementptr"(%6, %0) <{elem_type = i64, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
  %8 = "llvm.getelementptr"(%4, %0) <{elem_type = i32, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
  "llvm.store"(%5, %6) <{alignment = 32 : i64, nontemporal, ordering = 0 : i64, volatile_}> : (i64, !llvm.ptr) -> ()
}) : () -> ()
