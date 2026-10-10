// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
builtin.module {
    %0 = pybytecode.const 0
    %1 = pybytecode.const 1
    %2 = pybytecode.binop "add" %0 %1
}

// CHECK:       builtin.module {
// CHECK-NEXT:      %0 = pybytecode.const 0
// CHECK-NEXT:      %1 = pybytecode.const 1
// CHECK-NEXT:      %2 = pybytecode.binop "add" %0 %1
// CHECK-NEXT:  }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC:   %0 = "pybytecode.const"() <{const = 0 : i64}> : () -> !pybytecode.object
// CHECK-GENERIC:   %1 = "pybytecode.const"() <{const = 1 : i64}> : () -> !pybytecode.object
// CHECK-GENERIC:   %2 = "pybytecode.binop"(%0, %1) <{op = "add"}> : (!pybytecode.object, !pybytecode.object) -> !pybytecode.object
// CHECK-GENERIC: }) : () -> ()
