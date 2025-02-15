// RUN: xdsl-opt -p convert-func-to-x86-func --split-input-file  %s | filecheck %s

func.func @foo_int(%0: i32, %1: i32, %2: i32, %3: i32, %4: i32, %5: i32) -> i32 {
  %a = "test.op"(%0,%1): (i32,i32) -> i32
  %b = "test.op" (%a,%2): (i32,i32) -> i32
  %c = "test.op" (%b,%3): (i32,i32) -> i32
  %d = "test.op" (%c,%4): (i32,i32) -> i32
  %e = "test.op" (%d,%5): (i32,i32) -> i32
  func.return %e: i32
}
// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @foo_int() {
// CHECK-NEXT:      %0 = x86.get_register : () -> !x86.reg<rdi>
// CHECK-NEXT:      %1 = x86.get_register : () -> !x86.reg<rsi>
// CHECK-NEXT:      %2 = x86.get_register : () -> !x86.reg<rdx>
// CHECK-NEXT:      %3 = x86.get_register : () -> !x86.reg<rcx>
// CHECK-NEXT:      %4 = x86.get_register : () -> !x86.reg<r8>
// CHECK-NEXT:      %5 = x86.get_register : () -> !x86.reg<r9>
// CHECK-NEXT:      %6 = builtin.unrealized_conversion_cast %0 : !x86.reg<rdi> to i32
// CHECK-NEXT:      %7 = builtin.unrealized_conversion_cast %1 : !x86.reg<rsi> to i32
// CHECK-NEXT:      %8 = builtin.unrealized_conversion_cast %2 : !x86.reg<rdx> to i32
// CHECK-NEXT:      %9 = builtin.unrealized_conversion_cast %3 : !x86.reg<rcx> to i32
// CHECK-NEXT:      %10 = builtin.unrealized_conversion_cast %4 : !x86.reg<r8> to i32
// CHECK-NEXT:      %11 = builtin.unrealized_conversion_cast %5 : !x86.reg<r9> to i32
// CHECK-NEXT:      %a = "test.op"(%0, %1) : (!x86.reg<rdi>, !x86.reg<rsi>) -> i32
// CHECK-NEXT:      %b = "test.op"(%a, %2) : (i32, !x86.reg<rdx>) -> i32
// CHECK-NEXT:      %c = "test.op"(%b, %3) : (i32, !x86.reg<rcx>) -> i32
// CHECK-NEXT:      %d = "test.op"(%c, %4) : (i32, !x86.reg<r8>) -> i32
// CHECK-NEXT:      %e = "test.op"(%d, %5) : (i32, !x86.reg<r9>) -> i32
// CHECK-NEXT:      %12 = builtin.unrealized_conversion_cast %e : i32 to !x86.reg
// CHECK-NEXT:      %13 = x86.get_register : () -> !x86.reg<rax>
// CHECK-NEXT:      %14 = x86.rr.mov %12, %13 : (!x86.reg, !x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  
