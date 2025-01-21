"builtin.module"() ({
  "builtin.module"() ({
    %47 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  }) : () -> ()
  "builtin.module"() ({
    "symref.declare"() {sym_name = "a"} : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "symref.declare"() {sym_name = "a"} : () -> ()
    %44 = "arith.constant"() <{value = 42 : i32}> : () -> i32
    "symref.update"(%44) {symbol = @a} : (i32) -> ()
    %45 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    "symref.update"(%44) {symbol = @a} : (i32) -> ()
    %46 = "arith.constant"() <{value = 23 : i32}> : () -> i32
    "symref.update"(%44) {symbol = @a} : (i32) -> ()
  }) : () -> ()
  "builtin.module"() ({
    "symref.declare"() {sym_name = "a"} : () -> ()
    %39 = "arith.constant"() <{value = 42 : i32}> : () -> i32
    "symref.update"(%39) {symbol = @a} : (i32) -> ()
    %40 = "symref.fetch"() {symbol = @a} : () -> i32
    %41 = "arith.addi"(%40, %40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %42 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %43 = "arith.muli"(%40, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  }) : () -> ()
  "builtin.module"() ({
    "symref.declare"() {sym_name = "a"} : () -> ()
    %34 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    "symref.update"(%34) {symbol = @a} : (i32) -> ()
    "symref.declare"() {sym_name = "b"} : () -> ()
    %35 = "arith.constant"() <{value = 22 : i32}> : () -> i32
    "symref.update"(%35) {symbol = @b} : (i32) -> ()
    %36 = "symref.fetch"() {symbol = @b} : () -> i32
    %37 = "symref.fetch"() {symbol = @a} : () -> i32
    %38 = "arith.addi"(%36, %37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  }) : () -> ()
  "builtin.module"() ({
    "symref.declare"() {sym_name = "a"} : () -> ()
    %23 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "symref.update"(%23) {symbol = @a} : (i32) -> ()
    "symref.declare"() {sym_name = "b"} : () -> ()
    %24 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "symref.update"(%24) {symbol = @b} : (i32) -> ()
    "symref.declare"() {sym_name = "c"} : () -> ()
    %25 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    "symref.update"(%25) {symbol = @c} : (i32) -> ()
    %26 = "symref.fetch"() {symbol = @b} : () -> i32
    %27 = "symref.fetch"() {symbol = @c} : () -> i32
    %28 = "arith.addi"(%26, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "symref.update"(%28) {symbol = @a} : (i32) -> ()
    %29 = "symref.fetch"() {symbol = @a} : () -> i32
    %30 = "symref.fetch"() {symbol = @b} : () -> i32
    %31 = "symref.fetch"() {symbol = @c} : () -> i32
    %32 = "arith.muli"(%29, %30) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "symref.update"(%32) {symbol = @b} : (i32) -> ()
    %33 = "arith.addi"(%32, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "symref.update"(%33) {symbol = @c} : (i32) -> ()
  }) : () -> ()
  "builtin.module"() ({
    %20 = "symref.fetch"() {symbol = @a} : () -> i32
    %21 = "symref.fetch"() {symbol = @b} : () -> i32
    %22 = "symref.fetch"() {symbol = @b} : () -> i32
  }) : () -> ()
  "builtin.module"() ({
    %17 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "symref.update"(%17) {symbol = @a} : (i32) -> ()
    %18 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "symref.update"(%18) {symbol = @a} : (i32) -> ()
    %19 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    "symref.update"(%19) {symbol = @c} : (i32) -> ()
    "symref.update"(%19) {symbol = @c} : (i32) -> ()
    "symref.update"(%19) {symbol = @c} : (i32) -> ()
  }) : () -> ()
  "builtin.module"() ({
    %12 = "symref.fetch"() {symbol = @b} : () -> i32
    %13 = "arith.muli"(%12, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %14 = "symref.fetch"() {symbol = @b} : () -> i32
    %15 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %16 = "arith.addi"(%14, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  }) : () -> ()
  "builtin.module"() ({
    %0 = "symref.fetch"() {symbol = @d} : () -> i32
    "symref.update"(%0) {symbol = @a} : (i32) -> ()
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "symref.update"(%1) {symbol = @b} : (i32) -> ()
    %2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    "symref.update"(%2) {symbol = @c} : (i32) -> ()
    %3 = "symref.fetch"() {symbol = @b} : () -> i32
    %4 = "symref.fetch"() {symbol = @c} : () -> i32
    %5 = "arith.addi"(%3, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "symref.update"(%5) {symbol = @a} : (i32) -> ()
    %6 = "symref.fetch"() {symbol = @a} : () -> i32
    %7 = "symref.fetch"() {symbol = @b} : () -> i32
    %8 = "arith.muli"(%6, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "symref.update"(%8) {symbol = @a} : (i32) -> ()
    %9 = "symref.fetch"() {symbol = @b} : () -> i32
    %10 = "symref.fetch"() {symbol = @c} : () -> i32
    %11 = "arith.addi"(%9, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "symref.update"(%11) {symbol = @c} : (i32) -> ()
  }) : () -> ()
}) : () -> ()
