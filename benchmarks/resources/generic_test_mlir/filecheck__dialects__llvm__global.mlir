"builtin.module"() ({
  "llvm.mlir.global"() <{addr_space = 0 : i32, constant, global_type = !llvm.array<13 x i8>, linkage = #llvm.linkage<internal>, sym_name = "str0", unnamed_addr = 0 : i64, value = "Hello world!\0A", visibility_ = 0 : i64}> ({
  }) : () -> ()
  %0 = "llvm.mlir.addressof"() <{global_name = @str0}> : () -> !llvm.ptr
  %1 = "llvm.getelementptr"(%0) <{elem_type = !llvm.array<13 x i8>, rawConstantIndices = array<i32: 0, 0>}> : (!llvm.ptr) -> !llvm.ptr
}) : () -> ()
