// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  "llvm.func"() <{
    sym_name = "empty",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
    "llvm.return"() : () -> ()
  }) : () -> ()
  // CHECK: define void @"empty"()
  // CHECK-NEXT: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "int_binops",
    function_type = !llvm.func<void (i32, i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0: i32, %arg1: i32):
    // CHECK: define void @"int_binops"(i32 %"{{.*}}", i32 %"{{.*}}")
    // CHECK-NEXT: {
    // CHECK-NEXT: entry:

    %0 = "llvm.add"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = add i32 %"{{.*}}", %"{{.*}}"
    %1 = "llvm.sub"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = sub i32 %"{{.*}}", %"{{.*}}"
    %2 = "llvm.mul"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = mul i32 %"{{.*}}", %"{{.*}}"
    %3 = "llvm.udiv"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = udiv i32 %"{{.*}}", %"{{.*}}"
    %4 = "llvm.sdiv"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = sdiv i32 %"{{.*}}", %"{{.*}}"
    %5 = "llvm.urem"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = urem i32 %"{{.*}}", %"{{.*}}"
    %6 = "llvm.srem"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = srem i32 %"{{.*}}", %"{{.*}}"
    %7 = "llvm.shl"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = shl i32 %"{{.*}}", %"{{.*}}"
    %8 = "llvm.lshr"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = lshr i32 %"{{.*}}", %"{{.*}}"
    %9 = "llvm.ashr"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = ashr i32 %"{{.*}}", %"{{.*}}"
    %10 = "llvm.and"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = and i32 %"{{.*}}", %"{{.*}}"
    %11 = "llvm.or"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = or i32 %"{{.*}}", %"{{.*}}"
    %12 = "llvm.xor"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = xor i32 %"{{.*}}", %"{{.*}}"
    "llvm.return"() : () -> ()
    // CHECK-NEXT:   ret void
    // CHECK-NEXT: }
  }) : () -> ()

  "llvm.func"() <{
    sym_name = "float_binops",
    function_type = !llvm.func<void (f32, f32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0: f32, %arg1: f32):
    // CHECK: define void @"float_binops"(float %"{{.*}}", float %"{{.*}}")
    // CHECK-NEXT: {
    // CHECK-NEXT: entry:

    %0 = "llvm.fadd"(%arg0, %arg1) : (f32, f32) -> f32
    // CHECK-NEXT:   %"{{.*}}" = fadd float %"{{.*}}", %"{{.*}}"
    %1 = "llvm.fsub"(%arg0, %arg1) : (f32, f32) -> f32
    // CHECK-NEXT:   %"{{.*}}" = fsub float %"{{.*}}", %"{{.*}}"
    %2 = "llvm.fmul"(%arg0, %arg1) : (f32, f32) -> f32
    // CHECK-NEXT:   %"{{.*}}" = fmul float %"{{.*}}", %"{{.*}}"
    %3 = "llvm.fdiv"(%arg0, %arg1) : (f32, f32) -> f32
    // CHECK-NEXT:   %"{{.*}}" = fdiv float %"{{.*}}", %"{{.*}}"
    %4 = "llvm.frem"(%arg0, %arg1) : (f32, f32) -> f32
    // CHECK-NEXT:   %"{{.*}}" = frem float %"{{.*}}", %"{{.*}}"
    "llvm.return"() : () -> ()
    // CHECK-NEXT:   ret void
    // CHECK-NEXT: }
  }) : () -> ()

  "llvm.func"() <{
    sym_name = "cast_ops",
    function_type = !llvm.func<void (i32, i64, f32, f64, !llvm.ptr)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%i32: i32, %i64: i64, %f32: f32, %f64: f64, %ptr: !llvm.ptr):
    // CHECK: define void @"cast_ops"(i32 %"{{.*}}", i64 %"{{.*}}", float %"{{.*}}", double %"{{.*}}", ptr %"{{.*}}")
    // CHECK-NEXT: {
    // CHECK-NEXT: entry:

    %0 = "llvm.trunc"(%i64) : (i64) -> i32
    // CHECK-NEXT:   %"{{.*}}" = trunc i64 %"{{.*}}" to i32
    %1 = "llvm.zext"(%i32) : (i32) -> i64
    // CHECK-NEXT:   %"{{.*}}" = zext i32 %"{{.*}}" to i64
    %2 = "llvm.sext"(%i32) : (i32) -> i64
    // CHECK-NEXT:   %"{{.*}}" = sext i32 %"{{.*}}" to i64
    %3 = "llvm.ptrtoint"(%ptr) : (!llvm.ptr) -> i64
    // CHECK-NEXT:   %"{{.*}}" = ptrtoint ptr %"{{.*}}" to i64
    %4 = "llvm.inttoptr"(%i64) : (i64) -> !llvm.ptr
    // CHECK-NEXT:   %"{{.*}}" = inttoptr i64 %"{{.*}}" to ptr
    %5 = "llvm.bitcast"(%i64) : (i64) -> f64
    // CHECK-NEXT:   %"{{.*}}" = bitcast i64 %"{{.*}}" to double
    %6 = "llvm.fpext"(%f32) : (f32) -> f64
    // CHECK-NEXT:   %"{{.*}}" = fpext float %"{{.*}}" to double

    "llvm.return"() : () -> ()
    // CHECK-NEXT:   ret void
    // CHECK-NEXT: }
  }) : () -> ()

  "llvm.func"() <{
    sym_name = "memory_ops",
    function_type = !llvm.func<void (i32, !llvm.ptr)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0: i32, %ptr: !llvm.ptr):
    // CHECK: define void @"memory_ops"(i32 %"{{.*}}", ptr %"{{.*}}")
    // CHECK-NEXT: {
    // CHECK-NEXT: entry:

    %0 = "llvm.alloca"(%arg0) <{elem_type = i32}> : (i32) -> !llvm.ptr
    // CHECK-NEXT:   %"{{.*}}" = alloca i32, i32 %"{{.*}}"
    "llvm.store"(%arg0, %0) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
    // CHECK-NEXT:   store i32 %"{{.*}}", {{ptr|i32\*}} %"{{.*}}"
    %1 = "llvm.load"(%0) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
    // CHECK-NEXT:   %"{{.*}}" = load i32, {{ptr|i32\*}} %"{{.*}}"
    %2 = "llvm.getelementptr"(%ptr, %arg0) <{elem_type = i32, rawConstantIndices = array<i32: -2147483648, 1>, noWrapFlags = 0 : i32}> : (!llvm.ptr, i32) -> !llvm.ptr
    // CHECK-NEXT:   %"{{.*}}" = getelementptr i32, ptr %"{{.*}}", i32 %"{{.*}}", i32 1
    "llvm.return"() : () -> ()
    // CHECK-NEXT:   ret void
    // CHECK-NEXT: }
  }) : () -> ()

  "llvm.func"() <{
    sym_name = "aggregate_ops",
    function_type = !llvm.func<void (!llvm.struct<(i32, f32)>, i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%agg: !llvm.struct<(i32, f32)>, %val: i32):
    // CHECK: define void @"aggregate_ops"({i32, float} %"{{.*}}", i32 %"{{.*}}")
    // CHECK-NEXT: {
    // CHECK-NEXT: entry:
    
    %0 = "llvm.extractvalue"(%agg) <{position = array<i64: 0>}> : (!llvm.struct<(i32, f32)>) -> i32
    // CHECK-NEXT:   %"{{.*}}" = extractvalue {i32, float} %"{{.*}}", 0
    %1 = "llvm.insertvalue"(%agg, %val) <{position = array<i64: 0>}> : (!llvm.struct<(i32, f32)>, i32) -> !llvm.struct<(i32, f32)>
    // CHECK-NEXT:   %"{{.*}}" = insertvalue {i32, float} %"{{.*}}", i32 %"{{.*}}", 0
    "llvm.return"() : () -> ()
    // CHECK-NEXT:   ret void
    // CHECK-NEXT: }
  }) : () -> ()

  "llvm.func"() <{
    sym_name = "other_ops",
    function_type = !llvm.func<void (i32, i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0: i32, %arg1: i32):
    // CHECK: define void @"other_ops"(i32 %"{{.*}}", i32 %"{{.*}}")
    // CHECK-NEXT: {
    // CHECK-NEXT: entry:
    
    %0 = "llvm.icmp"(%arg0, %arg1) <{predicate = 0 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp eq i32 %"{{.*}}", %"{{.*}}"
    %1 = "llvm.icmp"(%arg0, %arg1) <{predicate = 1 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp ne i32 %"{{.*}}", %"{{.*}}"
    %2 = "llvm.icmp"(%arg0, %arg1) <{predicate = 2 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp slt i32 %"{{.*}}", %"{{.*}}"
    %3 = "llvm.icmp"(%arg0, %arg1) <{predicate = 3 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp sle i32 %"{{.*}}", %"{{.*}}"
    %4 = "llvm.icmp"(%arg0, %arg1) <{predicate = 4 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp sgt i32 %"{{.*}}", %"{{.*}}"
    %5 = "llvm.icmp"(%arg0, %arg1) <{predicate = 5 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp sge i32 %"{{.*}}", %"{{.*}}"
    %6 = "llvm.icmp"(%arg0, %arg1) <{predicate = 6 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp ult i32 %"{{.*}}", %"{{.*}}"
    %7 = "llvm.icmp"(%arg0, %arg1) <{predicate = 7 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp ule i32 %"{{.*}}", %"{{.*}}"
    %8 = "llvm.icmp"(%arg0, %arg1) <{predicate = 8 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp ugt i32 %"{{.*}}", %"{{.*}}"
    %9 = "llvm.icmp"(%arg0, %arg1) <{predicate = 9 : i64}> : (i32, i32) -> i1
    // CHECK-NEXT:   %"{{.*}}" = icmp uge i32 %"{{.*}}", %"{{.*}}"
    
    // Call
    "llvm.call"(%arg0, %arg1) <{
      callee = @int_binops, 
      fastmathFlags = #llvm.fastmath<none>, 
      operandSegmentSizes = array<i32: 2, 0>
    }> : (i32, i32) -> ()
    // CHECK-NEXT:   call void @"int_binops"(i32 %"{{.*}}", i32 %"{{.*}}")

    // Inline Asm
    %10 = "llvm.inline_asm"(%arg0) <{
      asm_string = "mov $0, $1",
      constraints = "=r,r",
      has_side_effects = unit
    }> : (i32) -> i32
    // CHECK-NEXT:   %"{{.*}}" = call i32 asm sideeffect "mov $0, $1", "=r,r"(i32 %"{{.*}}")
    
    "llvm.return"() : () -> ()
    // CHECK-NEXT:   ret void
    // CHECK-NEXT: }
  }) : () -> ()

  "llvm.func"() <{
    sym_name = "unreachable_op",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0:
    // CHECK: define void @"unreachable_op"()
    // CHECK-NEXT: {
    // CHECK-NEXT: entry:
    "llvm.unreachable"() : () -> ()
    // CHECK-NEXT:   unreachable
    // CHECK-NEXT: }
  }) : () -> ()

}
