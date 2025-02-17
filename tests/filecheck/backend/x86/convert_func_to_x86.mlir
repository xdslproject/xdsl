// RUN: xdsl-opt -p convert-func-to-x86-func --split-input-file  %s | filecheck %s

func.func @foo_int(%0: i32, %1: i32, %2: i32, %3: i32, %4: i32, %5: i32, %6: i32, %7: i32) -> i32 {
  %a = "test.op"(%0,%1): (i32,i32) -> i32
  %b = "test.op" (%a,%2): (i32,i32) -> i32
  %c = "test.op" (%b,%3): (i32,i32) -> i32
  %d = "test.op" (%c,%4): (i32,i32) -> i32
  %e = "test.op" (%d,%5): (i32,i32) -> i32
  %f = "test.op" (%e,%6): (i32,i32) -> i32
  %g = "test.op" (%f,%7): (i32,i32) -> i32
  func.return %g: i32
}
// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @foo_int() -> () {
// CHECK-NEXT:      %0 = x86.get_register : () -> !x86.reg<rdi>
// CHECK-NEXT:      %1 = x86.get_register : () -> !x86.reg<rsi>
// CHECK-NEXT:      %2 = x86.get_register : () -> !x86.reg<rdx>
// CHECK-NEXT:      %3 = x86.get_register : () -> !x86.reg<rcx>
// CHECK-NEXT:      %4 = x86.get_register : () -> !x86.reg<r8>
// CHECK-NEXT:      %5 = x86.get_register : () -> !x86.reg<r9>
// CHECK-NEXT:      %6 = x86.get_register : () -> !x86.reg<rsp>
// CHECK-NEXT:      %7 = x86.get_register : () -> !x86.reg
// CHECK-NEXT:      %8 = x86.rm.mov %7, %6, 8 : (!x86.reg, !x86.reg<rsp>) -> !x86.reg
// CHECK-NEXT:      %9 = x86.get_register : () -> !x86.reg
// CHECK-NEXT:      %10 = x86.rm.mov %9, %6, 16 : (!x86.reg, !x86.reg<rsp>) -> !x86.reg
// CHECK-NEXT:      %11 = builtin.unrealized_conversion_cast %0 : !x86.reg<rdi> to i32
// CHECK-NEXT:      %12 = builtin.unrealized_conversion_cast %1 : !x86.reg<rsi> to i32
// CHECK-NEXT:      %13 = builtin.unrealized_conversion_cast %2 : !x86.reg<rdx> to i32
// CHECK-NEXT:      %14 = builtin.unrealized_conversion_cast %3 : !x86.reg<rcx> to i32
// CHECK-NEXT:      %15 = builtin.unrealized_conversion_cast %4 : !x86.reg<r8> to i32
// CHECK-NEXT:      %16 = builtin.unrealized_conversion_cast %5 : !x86.reg<r9> to i32
// CHECK-NEXT:      %17 = builtin.unrealized_conversion_cast %8 : !x86.reg to i32
// CHECK-NEXT:      %18 = builtin.unrealized_conversion_cast %10 : !x86.reg to i32
// CHECK-NEXT:      %a = "test.op"(%11, %12) : (i32, i32) -> i32
// CHECK-NEXT:      %b = "test.op"(%a, %13) : (i32, i32) -> i32
// CHECK-NEXT:      %c = "test.op"(%b, %14) : (i32, i32) -> i32
// CHECK-NEXT:      %d = "test.op"(%c, %15) : (i32, i32) -> i32
// CHECK-NEXT:      %e = "test.op"(%d, %16) : (i32, i32) -> i32
// CHECK-NEXT:      %f = "test.op"(%e, %17) : (i32, i32) -> i32
// CHECK-NEXT:      %g = "test.op"(%f, %18) : (i32, i32) -> i32
// CHECK-NEXT:      %19 = builtin.unrealized_conversion_cast %g : i32 to !x86.reg
// CHECK-NEXT:      %20 = x86.get_register : () -> !x86.reg<rax>
// CHECK-NEXT:      %21 = x86.rr.mov %19, %20 : (!x86.reg, !x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
