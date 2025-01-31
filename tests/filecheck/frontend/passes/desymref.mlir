// RUN: xdsl-opt %s -p frontend-desymrefy | filecheck %s


// CHECK: builtin.module {

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 5 : i32
// CHECK-NEXT: }
builtin.module {
  %0 = arith.constant 5 : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT: }
builtin.module {
  "symref.declare"() {"sym_name" = "a"} : () -> ()
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 42 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 11 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 23 : i32
// CHECK-NEXT: }
builtin.module {
  "symref.declare"() {"sym_name" = "a"} : () -> ()
  %1 = arith.constant 42 : i32
  "symref.update"(%1) {"symbol" = @a} : (i32) -> ()
  %2 = arith.constant 11 : i32
  "symref.update"(%1) {"symbol" = @a} : (i32) -> ()
  %3 = arith.constant 23 : i32
  "symref.update"(%1) {"symbol" = @a} : (i32) -> ()
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 42 : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 7 : i32
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: }
builtin.module {
  "symref.declare"() {"sym_name" = "a"} : () -> ()
  %4 = arith.constant 42 : i32
  "symref.update"(%4) {"symbol" = @a} : (i32) -> ()
  %5 = "symref.fetch"() {"symbol" = @a} : () -> i32
  %6 = arith.addi %5, %5 : i32
  %7 = arith.constant 7 : i32
  %8 = arith.muli %5, %7 : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 11 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 22 : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: }
builtin.module {
  "symref.declare"() {"sym_name" = "a"} : () -> ()
  %9 = arith.constant 11 : i32
  "symref.update"(%9) {"symbol" = @a} : (i32) -> ()
  "symref.declare"() {"sym_name" = "b"} : () -> ()
  %10 = arith.constant 22 : i32
  "symref.update"(%10) {"symbol" = @b} : (i32) -> ()
  %11 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %12 = "symref.fetch"() {"symbol" = @a} : () -> i32
  %13 = arith.addi %11, %12 : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: }
builtin.module {
  "symref.declare"() {"sym_name" = "a"} : () -> ()
  %14 = arith.constant 0 : i32
  "symref.update"(%14) {"symbol" = @a} : (i32) -> ()
  "symref.declare"() {"sym_name" = "b"} : () -> ()
  %15 = arith.constant 1 : i32
  "symref.update"(%15) {"symbol" = @b} : (i32) -> ()
  "symref.declare"() {"sym_name" = "c"} : () -> ()
  %16 = arith.constant 2 : i32
  "symref.update"(%16) {"symbol" = @c} : (i32) -> ()
  %17 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %18 = "symref.fetch"() {"symbol" = @c} : () -> i32
  %19 = arith.addi %17, %18 : i32
  "symref.update"(%19) {"symbol" = @a} : (i32) -> ()
  %20 = "symref.fetch"() {"symbol" = @a} : () -> i32
  %21 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %22 = "symref.fetch"() {"symbol" = @c} : () -> i32
  %23 = arith.muli %20, %21 : i32
  "symref.update"(%23) {"symbol" = @b} : (i32) -> ()
  %24 = arith.addi %23, %22 : i32
  "symref.update"(%24) {"symbol" = @c} : (i32) -> ()
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT: }
builtin.module {
  %25 = "symref.fetch"() {"symbol" = @a} : () -> i32
  %26 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %27 = "symref.fetch"() {"symbol" = @b} : () -> i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:   "symref.update"(%{{.*}}) {symbol = @a} : (i32) -> ()
// CHECK-NEXT:   %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:   "symref.update"(%{{.*}}) {symbol = @c} : (i32) -> ()
// CHECK-NEXT: }
builtin.module {
  %28 = arith.constant 0 : i32
  "symref.update"(%28) {"symbol" = @a} : (i32) -> ()
  %29 = arith.constant 1 : i32
  "symref.update"(%29) {"symbol" = @a} : (i32) -> ()
  %30 = arith.constant 2 : i32
  "symref.update"(%30) {"symbol" = @c} : (i32) -> ()
  "symref.update"(%30) {"symbol" = @c} : (i32) -> ()
  "symref.update"(%30) {"symbol" = @c} : (i32) -> ()
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = "symref.fetch"() {symbol = @b} : () -> i32
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 5 : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: }
builtin.module {
  %31 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %32 = arith.muli %31, %31 : i32
  %33 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %34 = arith.constant 5 : i32
  %35 = arith.addi %33, %34 : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:   "symref.update"(%{{.*}}) {symbol = @b} : (i32) -> ()
// CHECK-NEXT:   %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   "symref.update"(%{{.*}}) {symbol = @a} : (i32) -> ()
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   "symref.update"(%{{.*}}) {symbol = @c} : (i32) -> ()
// CHECK-NEXT: }
builtin.module {
  %36 = "symref.fetch"() {"symbol" = @d} : () -> i32
  "symref.update"(%36) {"symbol" = @a} : (i32) -> ()
  %37 = arith.constant 1 : i32
  "symref.update"(%37) {"symbol" = @b} : (i32) -> ()
  %38 = arith.constant 2 : i32
  "symref.update"(%38) {"symbol" = @c} : (i32) -> ()
  %39 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %40 = "symref.fetch"() {"symbol" = @c} : () -> i32
  %41 = arith.addi %39, %40 : i32
  "symref.update"(%41) {"symbol" = @a} : (i32) -> ()
  %42 = "symref.fetch"() {"symbol" = @a} : () -> i32
  %43 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %44 = arith.muli %42, %43 : i32
  "symref.update"(%44) {"symbol" = @a} : (i32) -> ()
  %45 = "symref.fetch"() {"symbol" = @b} : () -> i32
  %46 = "symref.fetch"() {"symbol" = @c} : () -> i32
  %47 = arith.addi %45, %46 : i32
  "symref.update"(%47) {"symbol" = @c} : (i32) -> ()
}

// CHECK-NEXT: }
