// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
builtin.module {
    %0 = py.const 0
    %1 = py.const 1
    %2 = py.binop "add" %0 %1
}

// CHECK:       builtin.module {
// CHECK-NEXT:      %0 = py.const 0
// CHECK-NEXT:      %1 = py.const 1
// CHECK-NEXT:      %2 = py.binop "add" %0 %1
// CHECK-NEXT:  }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC:   %0 = "py.const"() <{const = 0 : i64}> : () -> !py.object
// CHECK-GENERIC:   %1 = "py.const"() <{const = 1 : i64}> : () -> !py.object
// CHECK-GENERIC:   %2 = "py.binop"(%0, %1) <{op = "add"}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC: }) : () -> ()
