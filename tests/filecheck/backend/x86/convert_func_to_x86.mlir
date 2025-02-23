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
// CHECK-NEXT:    x86_func.func @foo_int(%0 : !x86.reg<rdi>, %1 : !x86.reg<rsi>, %2 : !x86.reg<rdx>, %3 : !x86.reg<rcx>, %4 : !x86.reg<r8>, %5 : !x86.reg<r9>, %6 : !x86.reg<rsp>) -> !x86.reg<rax> {
// CHECK-NEXT:      %7 = builtin.unrealized_conversion_cast %0 : !x86.reg<rdi> to i32
// CHECK-NEXT:      %8 = builtin.unrealized_conversion_cast %1 : !x86.reg<rsi> to i32
// CHECK-NEXT:      %9 = builtin.unrealized_conversion_cast %2 : !x86.reg<rdx> to i32
// CHECK-NEXT:      %10 = builtin.unrealized_conversion_cast %3 : !x86.reg<rcx> to i32
// CHECK-NEXT:      %11 = builtin.unrealized_conversion_cast %4 : !x86.reg<r8> to i32
// CHECK-NEXT:      %12 = builtin.unrealized_conversion_cast %5 : !x86.reg<r9> to i32
// CHECK-NEXT:      %13 = x86.get_register : () -> !x86.reg
// CHECK-NEXT:      %14 = x86.rm.mov %13, %6, 8 : (!x86.reg, !x86.reg<rsp>) -> !x86.reg
// CHECK-NEXT:      %15 = builtin.unrealized_conversion_cast %14 : !x86.reg to i32
// CHECK-NEXT:      %16 = x86.get_register : () -> !x86.reg
// CHECK-NEXT:      %17 = x86.rm.mov %16, %6, 16 : (!x86.reg, !x86.reg<rsp>) -> !x86.reg
// CHECK-NEXT:      %18 = builtin.unrealized_conversion_cast %17 : !x86.reg to i32
// CHECK-NEXT:      %a = "test.op"(%7, %8) : (i32, i32) -> i32
// CHECK-NEXT:      %b = "test.op"(%a, %9) : (i32, i32) -> i32
// CHECK-NEXT:      %c = "test.op"(%b, %10) : (i32, i32) -> i32
// CHECK-NEXT:      %d = "test.op"(%c, %11) : (i32, i32) -> i32
// CHECK-NEXT:      %e = "test.op"(%d, %12) : (i32, i32) -> i32
// CHECK-NEXT:      %f = "test.op"(%e, %15) : (i32, i32) -> i32
// CHECK-NEXT:      %g = "test.op"(%f, %18) : (i32, i32) -> i32
// CHECK-NEXT:      %19 = builtin.unrealized_conversion_cast %g : i32 to !x86.reg
// CHECK-NEXT:      %20 = x86.get_register : () -> !x86.reg<rax>
// CHECK-NEXT:      %21 = x86.rr.mov %19, %20 : (!x86.reg, !x86.reg<rax>) -> !x86.reg<rax>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
