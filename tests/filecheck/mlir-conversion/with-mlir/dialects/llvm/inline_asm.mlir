/// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

%0 = "test.op"() : () -> i32
%1 = "test.op"() : () -> i32
"llvm.inline_asm"(%0, %1) <{asm_string = "csrw $0, $1", constraints = "i, r"}> {has_side_effect} : (i32, i32) -> ()

%2 = "test.op"() : () -> vector<8xf32>
%3 = "test.op"() : () -> vector<8xf32>
%4 = "llvm.inline_asm"(%2, %3) <{asm_dialect = 1 : i64, asm_string = "vaddps $0, $1, $2", constraints = "=x,x,x"}> : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>

// CHECK: %0 = "test.op"() : () -> i32
// CHECK-NEXT: 1 = "test.op"() : () -> i32
// CHECK-NEXT: "llvm.inline_asm"(%0, %1) <{"asm_string" = "csrw $0, $1", "constraints" = "i, r"}> {"has_side_effect"} : (i32, i32) -> ()
// CHECK-NEXT: %2 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT: %3 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT: %4 = "llvm.inline_asm"(%2, %3) <{"asm_dialect" = 1 : i64, "asm_string" = "vaddps $0, $1, $2", "constraints" = "=x,x,x"}> : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
