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
  symref.declare "a"
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 42 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 11 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 23 : i32
// CHECK-NEXT: }
builtin.module {
  symref.declare "a"
  %1 = arith.constant 42 : i32
  symref.update @a = %1 : i32
  %2 = arith.constant 11 : i32
  symref.update @a = %1 : i32
  %3 = arith.constant 23 : i32
  symref.update @a = %1 : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 42 : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 7 : i32
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: }
builtin.module {
  symref.declare "a"
  %4 = arith.constant 42 : i32
  symref.update @a = %4 : i32
  %5 = symref.fetch @a : i32
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
  symref.declare "a"
  %9 = arith.constant 11 : i32
  symref.update @a = %9 : i32
  symref.declare "b"
  %10 = arith.constant 22 : i32
  symref.update @b = %10 : i32
  %11 = symref.fetch @b : i32
  %12 = symref.fetch @a : i32
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
  symref.declare "a"
  %14 = arith.constant 0 : i32
  symref.update @a = %14 : i32
  symref.declare "b"
  %15 = arith.constant 1 : i32
  symref.update @b = %15 : i32
  symref.declare "c"
  %16 = arith.constant 2 : i32
  symref.update @c = %16 : i32
  %17 = symref.fetch @b : i32
  %18 = symref.fetch @c : i32
  %19 = arith.addi %17, %18 : i32
  symref.update @a = %19 : i32
  %20 = symref.fetch @a : i32
  %21 = symref.fetch @b : i32
  %22 = symref.fetch @c : i32
  %23 = arith.muli %20, %21 : i32
  symref.update @b = %23 : i32
  %24 = arith.addi %23, %22 : i32
  symref.update @c = %24 : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT: }
builtin.module {
  %25 = symref.fetch @a : i32
  %26 = symref.fetch @b : i32
  %27 = symref.fetch @b : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:   symref.update @a = %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:   symref.update @c = %{{.*}} : i32
// CHECK-NEXT: }
builtin.module {
  %28 = arith.constant 0 : i32
  symref.update @a = %28 : i32
  %29 = arith.constant 1 : i32
  symref.update @a = %29 : i32
  %30 = arith.constant 2 : i32
  symref.update @c = %30 : i32
  symref.update @c = %30 : i32
  symref.update @c = %30 : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = symref.fetch @b : i32
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 5 : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: }
builtin.module {
  %31 = symref.fetch @b : i32
  %32 = arith.muli %31, %31 : i32
  %33 = symref.fetch @b : i32
  %34 = arith.constant 5 : i32
  %35 = arith.addi %33, %34 : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:   symref.update @b = %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   symref.update @a = %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   symref.update @c = %{{.*}} : i32
// CHECK-NEXT: }
builtin.module {
  %36 = symref.fetch @d : i32
  symref.update @a = %36 : i32
  %37 = arith.constant 1 : i32
  symref.update @b = %37 : i32
  %38 = arith.constant 2 : i32
  symref.update @c = %38 : i32
  %39 = symref.fetch @b : i32
  %40 = symref.fetch @c : i32
  %41 = arith.addi %39, %40 : i32
  symref.update @a = %41 : i32
  %42 = symref.fetch @a : i32
  %43 = symref.fetch @b : i32
  %44 = arith.muli %42, %43 : i32
  symref.update @a = %44 : i32
  %45 = symref.fetch @b : i32
  %46 = symref.fetch @c : i32
  %47 = arith.addi %45, %46 : i32
  symref.update @c = %47 : i32
}


// Reads inside an scf.if see the value from before it.

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %cond = "test.op"() : () -> i1
// CHECK-NEXT:   %one = arith.constant 1 : i32
// CHECK-NEXT:   %r = scf.if %cond -> (i32) {
// CHECK-NEXT:     scf.yield %one : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %sum = arith.addi %one, %one : i32
// CHECK-NEXT:     scf.yield %sum : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%r) : (i32) -> ()
// CHECK-NEXT: }
builtin.module {
  %cond = "test.op"() : () -> i1
  symref.declare "a"
  %one = arith.constant 1 : i32
  symref.update @a = %one : i32
  %r = scf.if %cond -> (i32) {
    %a1 = symref.fetch @a : i32
    scf.yield %a1 : i32
  } else {
    %a2 = symref.fetch @a : i32
    %sum = arith.addi %a2, %a2 : i32
    scf.yield %sum : i32
  }
  "test.op"(%r) : (i32) -> ()
}


// A value written in one branch is yielded, and the other branch yields the
// value from before.

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %cond = "test.op"() : () -> i1
// CHECK-NEXT:   %one = arith.constant 1 : i32
// CHECK-NEXT:   %a2 = scf.if %cond -> (i32) {
// CHECK-NEXT:     %two = arith.addi %one, %one : i32
// CHECK-NEXT:     scf.yield %two : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %one : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%a2) : (i32) -> ()
// CHECK-NEXT: }
builtin.module {
  %cond = "test.op"() : () -> i1
  symref.declare "a"
  %one = arith.constant 1 : i32
  symref.update @a = %one : i32
  scf.if %cond {
    %a1 = symref.fetch @a : i32
    %two = arith.addi %a1, %a1 : i32
    symref.update @a = %two : i32
    scf.yield
  } else {
    scf.yield
  }
  %a2 = symref.fetch @a : i32
  "test.op"(%a2) : (i32) -> ()
}


// Written values are yielded after the existing results.

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %cond = "test.op"() : () -> i1
// CHECK-NEXT:   %one = arith.constant 1 : i32
// CHECK-NEXT:   %r, %r_1, %r_2 = scf.if %cond -> (i32, i32, i32) {
// CHECK-NEXT:     %two = arith.constant 2 : i32
// CHECK-NEXT:     scf.yield %two, %two, %one : i32, i32, i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %three = arith.constant 3 : i32
// CHECK-NEXT:     scf.yield %three, %three, %three : i32, i32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   %b = scf.if %cond -> (i32) {
// CHECK-NEXT:     %four = arith.constant 4 : i32
// CHECK-NEXT:     scf.yield %four : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %r_2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%r, %r_1, %b) : (i32, i32, i32) -> ()
// CHECK-NEXT: }
builtin.module {
  %cond = "test.op"() : () -> i1
  symref.declare "a"
  symref.declare "b"
  %one = arith.constant 1 : i32
  symref.update @a = %one : i32
  symref.update @b = %one : i32
  %r = scf.if %cond -> (i32) {
    %two = arith.constant 2 : i32
    symref.update @a = %two : i32
    scf.yield %two : i32
  } else {
    %three = arith.constant 3 : i32
    symref.update @a = %three : i32
    symref.update @b = %three : i32
    scf.yield %three : i32
  }
  scf.if %cond {
    %four = arith.constant 4 : i32
    symref.update @b = %four : i32
    scf.yield
  } else {
    scf.yield
  }
  %a = symref.fetch @a : i32
  %b = symref.fetch @b : i32
  "test.op"(%r, %a, %b) : (i32, i32, i32) -> ()
}


// Nested scf.if operations are promoted from the inside out, and symbols
// declared in a branch stay in it.

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %cond = "test.op"() : () -> i1
// CHECK-NEXT:   %one = arith.constant 1 : i32
// CHECK-NEXT:   %a2 = scf.if %cond -> (i32) {
// CHECK-NEXT:     %0 = scf.if %cond -> (i32) {
// CHECK-NEXT:       %two = arith.addi %one, %one : i32
// CHECK-NEXT:       scf.yield %two : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %one : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %0 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %one : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%a2) : (i32) -> ()
// CHECK-NEXT: }
builtin.module {
  %cond = "test.op"() : () -> i1
  symref.declare "a"
  %one = arith.constant 1 : i32
  symref.update @a = %one : i32
  scf.if %cond {
    symref.declare "b"
    %a1 = symref.fetch @a : i32
    symref.update @b = %a1 : i32
    scf.if %cond {
      %b = symref.fetch @b : i32
      %two = arith.addi %b, %b : i32
      symref.update @a = %two : i32
      scf.yield
    } else {
      scf.yield
    }
    scf.yield
  } else {
    scf.yield
  }
  %a2 = symref.fetch @a : i32
  "test.op"(%a2) : (i32) -> ()
}


// Other operations with regions are kept when they have no symbols.

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     %zero = arith.constant 0 : i32
// CHECK-NEXT:     "test.termop"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
builtin.module {
  "test.op"() ({
    %zero = arith.constant 0 : i32
    "test.termop"() : () -> ()
  }) : () -> ()
}

// CHECK-NEXT: }
