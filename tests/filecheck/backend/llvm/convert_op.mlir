// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  llvm.func @declaration()

  // CHECK: declare void @"declaration"()

  llvm.func @named_entry() {
  ^entry:
    llvm.return
  }

  // CHECK: define void @"named_entry"()
  // CHECK-NEXT: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @custom_name() {
  ^my_block:
    llvm.return
  }

  // CHECK: define void @"custom_name"()
  // CHECK-NEXT: {
  // CHECK-NEXT: my_block:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @return_void() {
    llvm.return
  }

  // CHECK: define void @"return_void"()
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @return_arg(%arg0: i32) -> i32 {
    llvm.return %arg0 : i32
  }

  // CHECK: define i32 @"return_arg"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret i32 %".1"
  // CHECK-NEXT: }

  llvm.func @return_second_arg(%arg0: i32, %arg1: i32) -> i32 {
    llvm.return %arg1 : i32
  }

  // CHECK: define i32 @"return_second_arg"(i32 %".1", i32 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret i32 %".2"
  // CHECK-NEXT: }

  llvm.func @binops(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32) {
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
    llvm.return
  }

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

  llvm.func @binops_flags(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32) {

    // CHECK: define void @"binops_flags"(i32 %".1", i32 %".2", float %".3", float %".4")
    // CHECK-NEXT: {
    // CHECK-NEXT: {{.[0-9]+}}:

    %add_none = llvm.add %arg0, %arg1 : i32
    %add_nsw = llvm.add %arg0, %arg1 overflow<nsw> : i32
    %add_nuw = llvm.add %arg0, %arg1 overflow<nuw> : i32
    %add_nsw_nuw = llvm.add %arg0, %arg1 overflow<nsw, nuw> : i32

    // CHECK-NEXT:   {{%.+}} = add i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = add nsw i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = add nuw i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = add {{(nsw nuw|nuw nsw)}} i32 %".1", %".2"

    %sub_none = llvm.sub %arg0, %arg1 : i32
    %sub_nsw = llvm.sub %arg0, %arg1 overflow<nsw> : i32
    %sub_nuw = llvm.sub %arg0, %arg1 overflow<nuw> : i32
    %sub_nsw_nuw = llvm.sub %arg0, %arg1 overflow<nsw, nuw> : i32

    // CHECK-NEXT:   {{%.+}} = sub i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = sub nsw i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = sub nuw i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = sub {{(nsw nuw|nuw nsw)}} i32 %".1", %".2"

    %mul_none = llvm.mul %arg0, %arg1 : i32
    %mul_nsw = llvm.mul %arg0, %arg1 overflow<nsw> : i32
    %mul_nuw = llvm.mul %arg0, %arg1 overflow<nuw> : i32
    %mul_nsw_nuw = llvm.mul %arg0, %arg1 overflow<nsw, nuw> : i32

    // CHECK-NEXT:   {{%.+}} = mul i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = mul nsw i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = mul nuw i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = mul {{(nsw nuw|nuw nsw)}} i32 %".1", %".2"

    %shl_none = llvm.shl %arg0, %arg1 : i32
    %shl_nsw = llvm.shl %arg0, %arg1 overflow<nsw> : i32
    %shl_nuw = llvm.shl %arg0, %arg1 overflow<nuw> : i32
    %shl_nsw_nuw = llvm.shl %arg0, %arg1 overflow<nsw, nuw> : i32

    // CHECK-NEXT:   {{%.+}} = shl i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = shl nsw i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = shl nuw i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = shl {{(nsw nuw|nuw nsw)}} i32 %".1", %".2"

    %udiv_none = llvm.udiv %arg0, %arg1 : i32
    %udiv_exact = llvm.udiv exact %arg0, %arg1 : i32

    // CHECK-NEXT:   {{%.+}} = udiv i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = udiv exact i32 %".1", %".2"

    %sdiv_none = llvm.sdiv %arg0, %arg1 : i32
    %sdiv_exact = llvm.sdiv exact %arg0, %arg1 : i32

    // CHECK-NEXT:   {{%.+}} = sdiv i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = sdiv exact i32 %".1", %".2"

    %lshr_none = llvm.lshr %arg0, %arg1 : i32
    %lshr_exact = llvm.lshr exact %arg0, %arg1 : i32

    // CHECK-NEXT:   {{%.+}} = lshr i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = lshr exact i32 %".1", %".2"

    %ashr_none = llvm.ashr %arg0, %arg1 : i32
    %ashr_exact = llvm.ashr exact %arg0, %arg1 : i32

    // CHECK-NEXT:   {{%.+}} = ashr i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = ashr exact i32 %".1", %".2"

    %or_none = llvm.or %arg0, %arg1 : i32
    %or_disjoint = llvm.or disjoint %arg0, %arg1 : i32

    // CHECK-NEXT:   {{%.+}} = or i32 %".1", %".2"
    // CHECK-NEXT:   {{%.+}} = or disjoint i32 %".1", %".2"

    %fadd_none = llvm.fadd %arg2, %arg3 : f32
    %fadd_reassoc = llvm.fadd %arg2, %arg3 {fastmathFlags = #llvm.fastmath<reassoc>} : f32
    %fadd_nnan = llvm.fadd %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nnan>} : f32
    %fadd_ninf = llvm.fadd %arg2, %arg3 {fastmathFlags = #llvm.fastmath<ninf>} : f32
    %fadd_nsz = llvm.fadd %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nsz>} : f32
    %fadd_arcp = llvm.fadd %arg2, %arg3 {fastmathFlags = #llvm.fastmath<arcp>} : f32
    %fadd_contract = llvm.fadd %arg2, %arg3 {fastmathFlags = #llvm.fastmath<contract>} : f32
    %fadd_afn = llvm.fadd %arg2, %arg3 {fastmathFlags = #llvm.fastmath<afn>} : f32
    %fadd_fast = llvm.fadd %arg2, %arg3 {fastmathFlags = #llvm.fastmath<fast>} : f32

    // CHECK-NEXT:   {{%.+}} = fadd float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fadd reassoc float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fadd nnan float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fadd ninf float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fadd nsz float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fadd arcp float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fadd contract float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fadd afn float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fadd {{.*}} float %".3", %".4"

    %fsub_none = llvm.fsub %arg2, %arg3 : f32
    %fsub_reassoc = llvm.fsub %arg2, %arg3 {fastmathFlags = #llvm.fastmath<reassoc>} : f32
    %fsub_nnan = llvm.fsub %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nnan>} : f32
    %fsub_ninf = llvm.fsub %arg2, %arg3 {fastmathFlags = #llvm.fastmath<ninf>} : f32
    %fsub_nsz = llvm.fsub %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nsz>} : f32
    %fsub_arcp = llvm.fsub %arg2, %arg3 {fastmathFlags = #llvm.fastmath<arcp>} : f32
    %fsub_contract = llvm.fsub %arg2, %arg3 {fastmathFlags = #llvm.fastmath<contract>} : f32
    %fsub_afn = llvm.fsub %arg2, %arg3 {fastmathFlags = #llvm.fastmath<afn>} : f32
    %fsub_fast = llvm.fsub %arg2, %arg3 {fastmathFlags = #llvm.fastmath<fast>} : f32

    // CHECK-NEXT:   {{%.+}} = fsub float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fsub reassoc float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fsub nnan float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fsub ninf float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fsub nsz float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fsub arcp float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fsub contract float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fsub afn float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fsub {{.*}} float %".3", %".4"

    %fmul_none = llvm.fmul %arg2, %arg3 : f32
    %fmul_reassoc = llvm.fmul %arg2, %arg3 {fastmathFlags = #llvm.fastmath<reassoc>} : f32
    %fmul_nnan = llvm.fmul %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nnan>} : f32
    %fmul_ninf = llvm.fmul %arg2, %arg3 {fastmathFlags = #llvm.fastmath<ninf>} : f32
    %fmul_nsz = llvm.fmul %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nsz>} : f32
    %fmul_arcp = llvm.fmul %arg2, %arg3 {fastmathFlags = #llvm.fastmath<arcp>} : f32
    %fmul_contract = llvm.fmul %arg2, %arg3 {fastmathFlags = #llvm.fastmath<contract>} : f32
    %fmul_afn = llvm.fmul %arg2, %arg3 {fastmathFlags = #llvm.fastmath<afn>} : f32
    %fmul_fast = llvm.fmul %arg2, %arg3 {fastmathFlags = #llvm.fastmath<fast>} : f32

    // CHECK-NEXT:   {{%.+}} = fmul float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fmul reassoc float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fmul nnan float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fmul ninf float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fmul nsz float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fmul arcp float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fmul contract float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fmul afn float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fmul {{.*}} float %".3", %".4"

    %fdiv_none = llvm.fdiv %arg2, %arg3 : f32
    %fdiv_reassoc = llvm.fdiv %arg2, %arg3 {fastmathFlags = #llvm.fastmath<reassoc>} : f32
    %fdiv_nnan = llvm.fdiv %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nnan>} : f32
    %fdiv_ninf = llvm.fdiv %arg2, %arg3 {fastmathFlags = #llvm.fastmath<ninf>} : f32
    %fdiv_nsz = llvm.fdiv %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nsz>} : f32
    %fdiv_arcp = llvm.fdiv %arg2, %arg3 {fastmathFlags = #llvm.fastmath<arcp>} : f32
    %fdiv_contract = llvm.fdiv %arg2, %arg3 {fastmathFlags = #llvm.fastmath<contract>} : f32
    %fdiv_afn = llvm.fdiv %arg2, %arg3 {fastmathFlags = #llvm.fastmath<afn>} : f32
    %fdiv_fast = llvm.fdiv %arg2, %arg3 {fastmathFlags = #llvm.fastmath<fast>} : f32

    // CHECK-NEXT:   {{%.+}} = fdiv float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fdiv reassoc float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fdiv nnan float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fdiv ninf float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fdiv nsz float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fdiv arcp float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fdiv contract float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fdiv afn float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = fdiv {{.*}} float %".3", %".4"

    %frem_none = llvm.frem %arg2, %arg3 : f32
    %frem_reassoc = llvm.frem %arg2, %arg3 {fastmathFlags = #llvm.fastmath<reassoc>} : f32
    %frem_nnan = llvm.frem %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nnan>} : f32
    %frem_ninf = llvm.frem %arg2, %arg3 {fastmathFlags = #llvm.fastmath<ninf>} : f32
    %frem_nsz = llvm.frem %arg2, %arg3 {fastmathFlags = #llvm.fastmath<nsz>} : f32
    %frem_arcp = llvm.frem %arg2, %arg3 {fastmathFlags = #llvm.fastmath<arcp>} : f32
    %frem_contract = llvm.frem %arg2, %arg3 {fastmathFlags = #llvm.fastmath<contract>} : f32
    %frem_afn = llvm.frem %arg2, %arg3 {fastmathFlags = #llvm.fastmath<afn>} : f32
    %frem_fast = llvm.frem %arg2, %arg3 {fastmathFlags = #llvm.fastmath<fast>} : f32

    // CHECK-NEXT:   {{%.+}} = frem float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = frem reassoc float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = frem nnan float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = frem ninf float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = frem nsz float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = frem arcp float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = frem contract float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = frem afn float %".3", %".4"
    // CHECK-NEXT:   {{%.+}} = frem {{.*}} float %".3", %".4"

    llvm.return
  }

  llvm.func @inline_asm(%arg0: i32) {
    "llvm.inline_asm"(%arg0) <{
      asm_string = "add $0, 1",
      constraints = "r",
      has_side_effects
    }> : (i32) -> ()
    llvm.return
  }

  // CHECK: define void @"inline_asm"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   call void asm sideeffect "add $0, 1", "r"(i32 %".1")
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @alloca_op(%arg0: i32) {
    %0 = "llvm.alloca"(%arg0) <{elem_type = i32}> : (i32) -> !llvm.ptr
    llvm.return
  }

  // CHECK: define void @"alloca_op"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = alloca i32, i32 %".1"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @alloca_op_with_alignment(%arg0: i32) {
    %0 = "llvm.alloca"(%arg0) <{alignment = 32 : i64, elem_type = i32}> : (i32) -> !llvm.ptr
    llvm.return
  }

  // CHECK: define void @"alloca_op_with_alignment"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = alloca i32, i32 %".1", align 32
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @load_op(%arg0: !llvm.ptr) {
    %0 = "llvm.load"(%arg0) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
    llvm.return
  }

  // CHECK: define void @"load_op"(ptr %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = load i32, ptr %".1"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @load_op_with_alignment(%arg0: !llvm.ptr) {
    %0 = "llvm.load"(%arg0) <{ordering = 0 : i64, alignment = 16 : i64}> : (!llvm.ptr) -> i32
    llvm.return
  }

  // CHECK: define void @"load_op_with_alignment"(ptr %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = load i32, ptr %".1", align 16
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @store_op(%arg0: i32, %arg1: !llvm.ptr) {
    "llvm.store"(%arg0, %arg1) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
    llvm.return
  }

  // CHECK: define void @"store_op"(i32 %".1", ptr %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   store i32 %".1", ptr %".2"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @store_op_with_alignment(%arg0: i32, %arg1: !llvm.ptr) {
    "llvm.store"(%arg0, %arg1) <{ordering = 0 : i64, alignment = 8 : i64}> : (i32, !llvm.ptr) -> ()
    llvm.return
  }

  // CHECK: define void @"store_op_with_alignment"(i32 %".1", ptr %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   store i32 %".1", ptr %".2", align 8
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @extract_op(%arg0: !llvm.struct<(i32, f32)>) {
    %0 = "llvm.extractvalue"(%arg0) <{position = array<i64: 0>}> : (!llvm.struct<(i32, f32)>) -> i32
    llvm.return
  }

  // CHECK: define void @"extract_op"({i32, float} %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = extractvalue {i32, float} %".1", 0
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @insert_op(%arg0: !llvm.struct<(i32, f32)>, %arg1: i32) {
    %0 = "llvm.insertvalue"(%arg0, %arg1) <{position = array<i64: 0>}> : (!llvm.struct<(i32, f32)>, i32) -> !llvm.struct<(i32, f32)>
    llvm.return
  }

  // CHECK: define void @"insert_op"({i32, float} %".1", i32 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = insertvalue {i32, float} %".1", i32 %".2", 0
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @unreachable_op() {
    llvm.unreachable
  }

  // CHECK: define void @"unreachable_op"()
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   unreachable
  // CHECK-NEXT: }

  llvm.func @casts(%arg0: i32, %arg1: i64, %arg2: !llvm.ptr, %arg3: f32) {
    %0 = llvm.trunc %arg1 : i64 to i32
    %1 = llvm.zext %arg0 : i32 to i64
    %2 = llvm.sext %arg0 : i32 to i64
    %3 = "llvm.ptrtoint"(%arg2) : (!llvm.ptr) -> i64
    %4 = "llvm.inttoptr"(%arg1) : (i64) -> !llvm.ptr
    %5 = llvm.bitcast %arg1 : i64 to f64
    %6 = llvm.fpext %arg3 : f32 to f64
    %7 = llvm.sitofp %arg0 : i32 to f32
    llvm.return
  }

  // CHECK: define void @"casts"(i32 %".1", i64 %".2", ptr %".3", float %".4")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = trunc i64 %".2" to i32
  // CHECK-NEXT:   {{%.+}} = zext i32 %".1" to i64
  // CHECK-NEXT:   {{%.+}} = sext i32 %".1" to i64
  // CHECK-NEXT:   {{%.+}} = ptrtoint ptr %".3" to i64
  // CHECK-NEXT:   {{%.+}} = inttoptr i64 %".2" to ptr
  // CHECK-NEXT:   {{%.+}} = bitcast i64 %".2" to double
  // CHECK-NEXT:   {{%.+}} = fpext float %".4" to double
  // CHECK-NEXT:   {{%.+}} = sitofp i32 %".1" to float
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @casts_with_flags(%arg0: i32, %arg1: i64) {
    // llvm.TruncOp with overflow flags (IntegerConversionOpOverflow)
    %trunc_none = llvm.trunc %arg1 : i64 to i32
    %trunc_nsw = llvm.trunc %arg1 overflow<nsw> : i64 to i32
    %trunc_nuw = llvm.trunc %arg1 overflow<nuw> : i64 to i32
    %trunc_both = llvm.trunc %arg1 overflow<nsw, nuw> : i64 to i32

    // llvm.ZExtOp with nneg flag (IntegerConversionOpNNeg)
    %zext_none = llvm.zext %arg0 : i32 to i64
    %zext_nneg = llvm.zext nneg %arg0 : i32 to i64

    llvm.return
  }

  // CHECK: define void @"casts_with_flags"(i32 %".1", i64 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = trunc i64 %".2" to i32
  // CHECK-NEXT:   {{%.+}} = trunc nsw i64 %".2" to i32
  // CHECK-NEXT:   {{%.+}} = trunc nuw i64 %".2" to i32
  // CHECK-NEXT:   {{%.+}} = trunc {{(nsw nuw|nuw nsw)}} i64 %".2" to i32
  // CHECK-NEXT:   {{%.+}} = zext i32 %".1" to i64
  // CHECK-NEXT:   {{%.+}} = zext nneg i32 %".1" to i64

  // void gep_constant(int (*ptr)[10]) {
  //   int* result = &ptr[1][2];
  // }
  llvm.func @gep_constant(%arg0: !llvm.ptr) {
    %0 = "llvm.getelementptr"(%arg0) <{
      elem_type = !llvm.array<10 x i32>,
      rawConstantIndices = array<i32: 1, 2>,
      noWrapFlags = 0 : i32
    }> : (!llvm.ptr) -> !llvm.ptr
    llvm.return
  }

  // CHECK: define void @"gep_constant"(ptr %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = getelementptr [10 x i32], ptr %".1", i32 1, i32 2
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  // void gep_ssa(int* ptr, int index) {
  //   int* result = &ptr[index];
  // }
  llvm.func @gep_ssa(%arg0: !llvm.ptr, %arg1: i32) {
    %0 = "llvm.getelementptr"(%arg0, %arg1) <{
      elem_type = i32,
      rawConstantIndices = array<i32: -2147483648>, // magic constant 0x80000000 (placeholder for ssa value)
      noWrapFlags = 0 : i32
    }> : (!llvm.ptr, i32) -> !llvm.ptr
    llvm.return
  }

  // CHECK: define void @"gep_ssa"(ptr %".1", i32 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = getelementptr i32, ptr %".1", i32 %".2"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  // void gep_mixed(ptr, int i, int j) {
  //   // e.g. ptr[1][i].field2[j]
  //   int* result = &ptr[1][i].field2[j];
  // }
  llvm.func @gep_mixed(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32) {
    %0 = "llvm.getelementptr"(%arg0, %arg1, %arg2) <{
      elem_type = !llvm.array<10 x !llvm.struct<(i32, i32, !llvm.array<10 x i32>)>>,
      rawConstantIndices = array<i32: 1, -2147483648, 2, -2147483648>,
      noWrapFlags = 0 : i32
    }> : (!llvm.ptr, i32, i32) -> !llvm.ptr
    llvm.return
  }

  // CHECK: define void @"gep_mixed"(ptr %".1", i32 %".2", i32 %".3")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = getelementptr [10 x {i32, i32, [10 x i32]}], ptr %".1", i32 1, i32 %".2", i32 2, i32 %".3"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  // void gep_inbounds(int* ptr, int idx) {
  //   // same as gep_ssa, but we assume that 'ptr + idx' stays within the same 'object'
  //   int* result = &ptr[idx]; 
  // }
  llvm.func @gep_inbounds(%arg0: !llvm.ptr, %arg1: i32) {
    %0 = "llvm.getelementptr"(%arg0, %arg1) <{
      elem_type = i32,
      rawConstantIndices = array<i32: -2147483648>,
      inbounds,
      noWrapFlags = 0 : i32
    }> : (!llvm.ptr, i32) -> !llvm.ptr
    llvm.return
  }

  // CHECK: define void @"gep_inbounds"(ptr %".1", i32 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = getelementptr inbounds i32, ptr %".1", i32 %".2"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @comparisons(%arg0: i32, %arg1: i32) {
    %0 = llvm.icmp "eq" %arg0, %arg1 : i32
    %1 = llvm.icmp "ne" %arg0, %arg1 : i32
    %2 = llvm.icmp "slt" %arg0, %arg1 : i32
    %3 = llvm.icmp "sle" %arg0, %arg1 : i32
    %4 = llvm.icmp "ult" %arg0, %arg1 : i32
    %5 = llvm.icmp "ule" %arg0, %arg1 : i32
    %6 = llvm.icmp "sgt" %arg0, %arg1 : i32
    %7 = llvm.icmp "sge" %arg0, %arg1 : i32
    %8 = llvm.icmp "ugt" %arg0, %arg1 : i32
    %9 = llvm.icmp "uge" %arg0, %arg1 : i32
    llvm.return
  }

  // CHECK: define void @"comparisons"(i32 %".1", i32 %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = icmp eq i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp ne i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp slt i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp sle i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp ult i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp ule i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp sgt i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp sge i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp ugt i32 %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = icmp uge i32 %".1", %".2"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @fcmp_op(%arg0: f32, %arg1: f32) {
    %0 = llvm.fcmp "oeq" %arg0, %arg1 : f32
    %1 = llvm.fcmp "ult" %arg0, %arg1 : f32
    llvm.return
  }

  // CHECK: define void @"fcmp_op"(float %".1", float %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = fcmp oeq float %".1", %".2"
  // CHECK-NEXT:   {{%.+}} = fcmp ult float %".1", %".2"
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @select_op(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
    %0 = llvm.select %arg0, %arg1, %arg2 : i1, i32
    llvm.return %0 : i32
  }

  // CHECK: define i32 @"select_op"(i1 %".1", i32 %".2", i32 %".3")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = select {{.*}}i1 %".1", i32 %".2", i32 %".3"
  // CHECK-NEXT:   ret i32 {{%.+}}
  // CHECK-NEXT: }

  llvm.func @br_op_no_args() {
    llvm.br ^bb1
  ^bb1:
    llvm.return
  }

  // CHECK: define void @"br_op_no_args"()
  // CHECK-NEXT: {
  // CHECK-NEXT: [[BB0:.\d+]]:
  // CHECK-NEXT:   br label %"[[BB1:.\d+]]"
  // CHECK-NEXT: [[BB1]]:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @br_op(%arg0: i32) -> i32 {
    llvm.br ^bb1(%arg0: i32)
  ^bb1(%0: i32):
    llvm.return %0 : i32
  }

  // CHECK: define i32 @"br_op"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[BB0:.\d+]]:
  // CHECK-NEXT:   br label %"[[BB1:.\d+]]"
  // CHECK-NEXT: [[BB1]]:
  // CHECK-NEXT:   %"[[V1:.\d+]]" = phi  i32 [%".1", %"[[BB0]]"]
  // CHECK-NEXT:   ret i32 %"[[V1]]"
  // CHECK-NEXT: }

  llvm.func @cond_br_op(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
    llvm.cond_br %arg0, ^bb1(%arg1: i32), ^bb2(%arg2: i32)
  ^bb1(%0: i32):
    llvm.return %0 : i32
  ^bb2(%1: i32):
    llvm.return %1 : i32
  }

  // CHECK: define i32 @"cond_br_op"(i1 %".1", i32 %".2", i32 %".3")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[BB0:.\d+]]:
  // CHECK-NEXT:   br i1 %".1", label %"[[BB1:.\d+]]", label %"[[BB2:.\d+]]"
  // CHECK-NEXT: [[BB1]]:
  // CHECK-NEXT:   %"[[V1:.\d+]]" = phi  i32 [%".2", %"[[BB0]]"]
  // CHECK-NEXT:   ret i32 %"[[V1]]"
  // CHECK-NEXT: [[BB2]]:
  // CHECK-NEXT:   %"[[V2:.\d+]]" = phi  i32 [%".3", %"[[BB0]]"]
  // CHECK-NEXT:   ret i32 %"[[V2]]"
  // CHECK-NEXT: }

  llvm.func @maxnum_op(%arg0: f32, %arg1: f32) -> f32 {
    %0 = llvm.intr.maxnum(%arg0, %arg1) : (f32, f32) -> f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"maxnum_op"(float %".1", float %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call float @"llvm.maxnum"(float %".1", float %".2")
  // CHECK-NEXT:   ret float %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @fabs_op(%arg0: f32) -> f32 {
    %0 = llvm.intr.fabs(%arg0) : (f32) -> f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"fabs_op"(float %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call float @"llvm.fabs"(float %".1")
  // CHECK-NEXT:   ret float %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @fsqrt_op(%arg0: f32) -> f32 {
    %0 = llvm.intr.sqrt(%arg0) : (f32) -> f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"fsqrt_op"(float %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call float @"llvm.sqrt"(float %".1")
  llvm.func @flog_op(%arg0: f32) -> f32 {
    %0 = llvm.intr.log(%arg0) : (f32) -> f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"flog_op"(float %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call float @"llvm.log"(float %".1")
  // CHECK-NEXT:   ret float %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @fneg_op(%arg0: f32) -> f32 {
    %0 = llvm.fneg %arg0 : f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"fneg_op"(float %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = fneg float %".1"
  // CHECK-NEXT:   ret float %"[[RES]]"
  // CHECK-NEXT: }

  // forward_ref_caller calls forward_ref_callee which is defined AFTER it
  llvm.func @forward_ref_caller(%arg0: i32) -> i32 {
    %0 = "llvm.call"(%arg0) <{
      callee = @forward_ref_callee,
      fastmathFlags = #llvm.fastmath<none>,
      CConv = #llvm.cconv<ccc>,
      TailCallKind = #llvm.tailcallkind<none>,
      operandSegmentSizes = array<i32: 1, 0>
    }> : (i32) -> i32
    llvm.return %0 : i32
  }

  // CHECK: define i32 @"forward_ref_caller"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call ccc i32 @"forward_ref_callee"(i32 %".1")
  // CHECK-NEXT:   ret i32 %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @forward_ref_callee(%arg0: i32) -> i32 {
    llvm.return %arg0 : i32
  }

  // CHECK: define i32 @"forward_ref_callee"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   ret i32 %".1"
  // CHECK-NEXT: }

  llvm.func @helper(%arg0: i32) -> i32 {
    llvm.return %arg0 : i32
  }

  // CHECK: define i32 @"helper"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   ret i32 %".1"
  // CHECK-NEXT: }

  llvm.func @call_op(%arg0: i32) -> i32 {
    %0 = "llvm.call"(%arg0) <{
      callee = @helper,
      fastmathFlags = #llvm.fastmath<nnan, ninf>,
      CConv = #llvm.cconv<fastcc>,
      TailCallKind = #llvm.tailcallkind<tail>,
      operandSegmentSizes = array<i32: 1, 0>
    }> : (i32) -> i32
    llvm.return %0 : i32
  }

  // CHECK: define i32 @"call_op"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = tail call ninf nnan fastcc i32 @"helper"(i32 %".1")
  // CHECK-NEXT:   ret i32 %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @masked_store_op(%arg0: vector<4xf32>, %arg1: !llvm.ptr, %arg2: vector<4xi1>) {
    llvm.intr.masked.store %arg0, %arg1, %arg2 {alignment = 16 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
    llvm.return
  }

  // CHECK: define void @"masked_store_op"(<4 x float> %".1", ptr %".2", <4 x i1> %".3")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   call void @"llvm.masked.store"(<4 x float> %".1", ptr %".2", i32 16, <4 x i1> %".3")
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @zero_op() -> !llvm.ptr {
    %0 = "llvm.mlir.zero"() : () -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }

  // CHECK: define ptr @"zero_op"()
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret ptr null
  // CHECK-NEXT: }

  llvm.func @fma_op_f32(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.fma %arg0, %arg1, %arg2 : vector<4xf32>
    llvm.return %0 : vector<4xf32>
  }

  // CHECK: define <4 x float> @"fma_op_f32"(<4 x float> %".1", <4 x float> %".2", <4 x float> %".3")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call <4 x float> @"llvm.fma.v4f32"(<4 x float> %".1", <4 x float> %".2", <4 x float> %".3")
  // CHECK-NEXT:   ret <4 x float> %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @fma_op_f64(%arg0: vector<2xf64>, %arg1: vector<2xf64>, %arg2: vector<2xf64>) -> vector<2xf64> {
    %0 = vector.fma %arg0, %arg1, %arg2 : vector<2xf64>
    llvm.return %0 : vector<2xf64>
  }

  // CHECK: define <2 x double> @"fma_op_f64"(<2 x double> %".1", <2 x double> %".2", <2 x double> %".3")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call <2 x double> @"llvm.fma.v2f64"(<2 x double> %".1", <2 x double> %".2", <2 x double> %".3")
  // CHECK-NEXT:   ret <2 x double> %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @callee(!llvm.ptr)

  llvm.func @addressof_target() {
    llvm.return
  }

  llvm.func @addressof_op() {
    %0 = "llvm.mlir.addressof"() <{global_name = @addressof_target}> : () -> !llvm.ptr
    "llvm.call"(%0) <{callee = @callee, fastmathFlags = #llvm.fastmath<>, CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, operandSegmentSizes = array<i32: 1, 0>}> : (!llvm.ptr) -> ()
    llvm.return
  }

  // CHECK: define void @"addressof_op"()
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   call ccc void @"callee"(void ()* @"addressof_target")
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  llvm.func @broadcast_f32(%arg0: f32) -> vector<4xf32> {
    %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
    llvm.return %0 : vector<4xf32>
  }

  // CHECK: define <4 x float> @"broadcast_f32"(float %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[INS:.\d+]]" = insertelement <4 x float> <float undef, float undef, float undef, float undef>, float %".1", i32 0
  // CHECK-NEXT:   %"[[RES:.\d+]]" = shufflevector <4 x float> %"[[INS]]", <4 x float> <float undef, float undef, float undef, float undef>, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  // CHECK-NEXT:   ret <4 x float> %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @constant_int() -> i32 {
    %0 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK: define i32 @"constant_int"()
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret i32 42
  // CHECK-NEXT: }

  llvm.func @constant_float() -> f32 {
    %0 = llvm.mlir.constant(3.14 : f32) : f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"constant_float"()
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret float 0x40091eb860000000
  // CHECK-NEXT: }

  llvm.func @constant_dense_vector() -> vector<4xi32> {
    %0 = llvm.mlir.constant(dense<[1, 2, 3, 4]> : vector<4xi32>) : vector<4xi32>
    llvm.return %0 : vector<4xi32>
  }

  // CHECK: define <4 x i32> @"constant_dense_vector"()
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  // CHECK-NEXT: }
}
