// RUN: xdsl-opt -p convert-func-to-x86-func --split-input-file  %s | filecheck %s

func.func @foo_const() {
  func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @foo_const(%0: !x86.reg64<rsp>) {
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func @foo_const() -> i32 {
  %0 = "test.op"(): () -> i32
  func.return %0: i32
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @foo_const(%0: !x86.reg64<rsp>) -> !x86.reg32<eax> {
// CHECK-NEXT:      %1 = "test.op"() : () -> i32
// CHECK-NEXT:      %2 = builtin.unrealized_conversion_cast %1 : i32 to !x86.reg32
// CHECK-NEXT:      %3 = x86.ds.mov %2 : (!x86.reg32) -> !x86.reg32<eax>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func public @foo_int(%0: i32, %1: i32, %2: i32, %3: i32, %4: i32, %5: i32, %6: i32, %7: i32) -> i32 {
  %a = "test.op"(%0,%1): (i32,i32) -> i32
  %b = "test.op" (%a,%2): (i32,i32) -> i32
  %c = "test.op" (%b,%3): (i32,i32) -> i32
  %d = "test.op" (%c,%4): (i32,i32) -> i32
  %e = "test.op" (%d,%5): (i32,i32) -> i32
  %f = "test.op" (%e,%6): (i32,i32) -> i32
  %g = "test.op" (%f,%7): (i32,i32) -> i32
  func.return %g: i32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   x86.directive ".global" "foo_int"
// CHECK-NEXT:   x86_func.func public @foo_int(%0: !x86.reg32<edi>, %1: !x86.reg32<esi>, %2: !x86.reg32<edx>, %3: !x86.reg32<ecx>, %4: !x86.reg32<r8d>, %5: !x86.reg32<r9d>, %6: !x86.reg64<rsp>) -> !x86.reg32<eax> {
// CHECK-NEXT:     %7 = x86.ds.mov %0 : (!x86.reg32<edi>) -> !x86.reg32
// CHECK-NEXT:     %8 = builtin.unrealized_conversion_cast %7 : !x86.reg32 to i32
// CHECK-NEXT:     %9 = x86.ds.mov %1 : (!x86.reg32<esi>) -> !x86.reg32
// CHECK-NEXT:     %10 = builtin.unrealized_conversion_cast %9 : !x86.reg32 to i32
// CHECK-NEXT:     %11 = x86.ds.mov %2 : (!x86.reg32<edx>) -> !x86.reg32
// CHECK-NEXT:     %12 = builtin.unrealized_conversion_cast %11 : !x86.reg32 to i32
// CHECK-NEXT:     %13 = x86.ds.mov %3 : (!x86.reg32<ecx>) -> !x86.reg32
// CHECK-NEXT:     %14 = builtin.unrealized_conversion_cast %13 : !x86.reg32 to i32
// CHECK-NEXT:     %15 = x86.ds.mov %4 : (!x86.reg32<r8d>) -> !x86.reg32
// CHECK-NEXT:     %16 = builtin.unrealized_conversion_cast %15 : !x86.reg32 to i32
// CHECK-NEXT:     %17 = x86.ds.mov %5 : (!x86.reg32<r9d>) -> !x86.reg32
// CHECK-NEXT:     %18 = builtin.unrealized_conversion_cast %17 : !x86.reg32 to i32
// CHECK-NEXT:     %19 = x86.dm.mov %6, 8 {comment = "Load the 7th argument of the function"} : (!x86.reg64<rsp>) -> !x86.reg32
// CHECK-NEXT:     %20 = builtin.unrealized_conversion_cast %19 : !x86.reg32 to i32
// CHECK-NEXT:     %21 = x86.dm.mov %6, 16 {comment = "Load the 8th argument of the function"} : (!x86.reg64<rsp>) -> !x86.reg32
// CHECK-NEXT:     %22 = builtin.unrealized_conversion_cast %21 : !x86.reg32 to i32
// CHECK-NEXT:     %a = "test.op"(%8, %10) : (i32, i32) -> i32
// CHECK-NEXT:     %b = "test.op"(%a, %12) : (i32, i32) -> i32
// CHECK-NEXT:     %c = "test.op"(%b, %14) : (i32, i32) -> i32
// CHECK-NEXT:     %d = "test.op"(%c, %16) : (i32, i32) -> i32
// CHECK-NEXT:     %e = "test.op"(%d, %18) : (i32, i32) -> i32
// CHECK-NEXT:     %f = "test.op"(%e, %20) : (i32, i32) -> i32
// CHECK-NEXT:     %g = "test.op"(%f, %22) : (i32, i32) -> i32
// CHECK-NEXT:     %23 = builtin.unrealized_conversion_cast %g : i32 to !x86.reg32
// CHECK-NEXT:     %24 = x86.ds.mov %23 : (!x86.reg32) -> !x86.reg32<eax>
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// CHECK-NEXT: }
