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
}
