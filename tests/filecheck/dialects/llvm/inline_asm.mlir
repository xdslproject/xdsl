// RUN: XDSL_ROUNDTRIP

%0 = "test.op"() : () -> i32
%1 = "test.op"() : () -> i32
llvm.inline_asm has_side_effects "csrw $0, $1", "i, r" %0, %1 : (i32, i32) -> ()

llvm.inline_asm "add $0, 1", "r" %0 : (i32) -> ()

%2 = "test.op"() : () -> vector<8xf32>
%3 = "test.op"() : () -> vector<8xf32>
%4 = llvm.inline_asm asm_dialect = att "vaddps $0, $1, $2", "=x,x,x" %2, %3 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
%5 = llvm.inline_asm asm_dialect = intel "vaddps $0, $1, $2", "=x,x,x" %2, %3 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>

%6 = llvm.inline_asm is_align_stack tail_call_kind = <tail> "foo", "=r,r" %0, %1 : (i32, i32) -> i32

// CHECK: %0 = "test.op"() : () -> i32
// CHECK-NEXT: %1 = "test.op"() : () -> i32
// CHECK-NEXT: llvm.inline_asm has_side_effects "csrw $0, $1", "i, r" %0, %1 : (i32, i32) -> ()
// CHECK-NEXT: llvm.inline_asm "add $0, 1", "r" %0 : (i32) -> ()
// CHECK-NEXT: %2 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT: %3 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT: %4 = llvm.inline_asm asm_dialect = att "vaddps $0, $1, $2", "=x,x,x" %2, %3 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-NEXT: %5 = llvm.inline_asm asm_dialect = intel "vaddps $0, $1, $2", "=x,x,x" %2, %3 : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK-NEXT: %6 = llvm.inline_asm is_align_stack tail_call_kind = <tail> "foo", "=r,r" %0, %1 : (i32, i32) -> i32
