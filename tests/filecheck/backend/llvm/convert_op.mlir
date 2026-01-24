// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  "llvm.func"() <{
    sym_name = "declaration",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  }) : () -> ()

  // CHECK: declare void @"declaration"()

  "llvm.func"() <{
    sym_name = "named_entry",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^entry:
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"named_entry"()
  // CHECK-NEXT: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "custom_name",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^my_block:
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"custom_name"()
  // CHECK-NEXT: {
  // CHECK-NEXT: my_block:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "return_void",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"return_void"()
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "return_arg",
    function_type = !llvm.func<i32 (i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0 : i32):
    "llvm.return"(%arg0) : (i32) -> ()
  }) : () -> ()

  // CHECK: define i32 @"return_arg"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret i32 %".1"
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "return_second_arg",
    function_type = !llvm.func<i32 (i32, i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0 : i32, %arg1 : i32):
    "llvm.return"(%arg1) : (i32) -> ()
  }) : () -> ()

  // CHECK: define i32 @"return_second_arg"(i32 %".1", i32 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret i32 %".2"
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "binops",
    function_type = !llvm.func<void (i32, i32, f32, f32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0 : i32, %arg1 : i32, %arg2 : f32, %arg3 : f32):
    %0 = llvm.add %arg0, %arg1 : i32
    %1 = llvm.fadd %arg2, %arg3 : f32
    %2 = llvm.sub %arg0, %arg1 : i32
    %3 = llvm.fsub %arg2, %arg3 : f32
    %4 = llvm.mul %arg0, %arg1 : i32
    %5 = llvm.fmul %arg2, %arg3 : f32
    %6 = llvm.udiv %arg0, %arg1 : i32
    %7 = llvm.sdiv %arg0, %arg1 : i32
    %8 = llvm.fdiv %arg2, %arg3 : f32
    %9 = llvm.urem %arg0, %arg1 : i32
    %10 = llvm.srem %arg0, %arg1 : i32
    %11 = llvm.frem %arg2, %arg3 : f32
    %12 = llvm.shl %arg0, %arg1 : i32
    %13 = llvm.lshr %arg0, %arg1 : i32
    %14 = llvm.ashr %arg0, %arg1 : i32
    %15 = llvm.and %arg0, %arg1 : i32
    %16 = llvm.or %arg0, %arg1 : i32
    %17 = llvm.xor %arg0, %arg1 : i32
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"binops"(i32 %".1", i32 %".2", float %".3", float %".4")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = add i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = fadd float %".3", %".4"
  // CHECK-NEXT:   {{%.+}} = sub i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = fsub float %".3", %".4"
  // CHECK-NEXT:   {{%.+}} = mul i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = fmul float %".3", %".4"
  // CHECK-NEXT:   {{%.+}} = udiv i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = sdiv i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = fdiv float %".3", %".4"
  // CHECK-NEXT:   {{%.+}} = urem i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = srem i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = frem float %".3", %".4"
  // CHECK-NEXT:   {{%.+}} = shl i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = lshr i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = ashr i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = and i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = or i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = xor i32 %".1", %".2"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  // void gep_constant(int* ptr) {
  //   int* result = &ptr[1][2];
  // }
  "llvm.func"() <{
    sym_name = "gep_constant",
    function_type = !llvm.func<void (!llvm.ptr)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0 : !llvm.ptr):
    %0 = "llvm.getelementptr"(%arg0) <{
      elem_type = i32,
      rawConstantIndices = array<i32: 1, 2>,
      noWrapFlags = 0 : i32
    }> : (!llvm.ptr) -> !llvm.ptr
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"gep_constant"(ptr %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = getelementptr i32, ptr %".1", i32 1, i32 2
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  // void gep_ssa(int* ptr, int index) {
  //   int* result = &ptr[index];
  // }
  "llvm.func"() <{
    sym_name = "gep_ssa",
    function_type = !llvm.func<void (!llvm.ptr, i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0 : !llvm.ptr, %arg1 : i32):
    %0 = "llvm.getelementptr"(%arg0, %arg1) <{
      elem_type = i32,
      rawConstantIndices = array<i32: -2147483648>, // magic constant 0x80000000 (placeholder for ssa value)
      noWrapFlags = 0 : i32
    }> : (!llvm.ptr, i32) -> !llvm.ptr
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"gep_ssa"(ptr %".1", i32 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = getelementptr i32, ptr %".1", i32 %".2"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  // void gep_mixed(int* ptr, int i, int j) {
  //   // e.g. ptr[1].some_array[i].some_struct[2].some_data[j]
  //   int* result = &ptr[1][i][2][j];
  // }
  "llvm.func"() <{
    sym_name = "gep_mixed",
    function_type = !llvm.func<void (!llvm.ptr, i32, i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0 : !llvm.ptr, %arg1 : i32, %arg2 : i32):
    %0 = "llvm.getelementptr"(%arg0, %arg1, %arg2) <{
      elem_type = i32,
      rawConstantIndices = array<i32: 1, -2147483648, 2, -2147483648>,
      noWrapFlags = 0 : i32
    }> : (!llvm.ptr, i32, i32) -> !llvm.ptr
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"gep_mixed"(ptr %".1", i32 %".2", i32 %".3")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = getelementptr i32, ptr %".1", i32 1, i32 %".2", i32 2, i32 %".3"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  // void gep_inbounds(int* ptr, int idx) {
  //   // same as gep_ssa, but we assume that 'ptr + idx' stays within the same 'object'
  //   int* result = &ptr[idx]; 
  // }
  "llvm.func"() <{
    sym_name = "gep_inbounds",
    function_type = !llvm.func<void (!llvm.ptr, i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0 : !llvm.ptr, %arg1 : i32):
    %0 = "llvm.getelementptr"(%arg0, %arg1) <{
      elem_type = i32,
      rawConstantIndices = array<i32: -2147483648>,
      inbounds,
      noWrapFlags = 0 : i32
    }> : (!llvm.ptr, i32) -> !llvm.ptr
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"gep_inbounds"(ptr %".1", i32 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = getelementptr inbounds i32, ptr %".1", i32 %".2"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }
}
