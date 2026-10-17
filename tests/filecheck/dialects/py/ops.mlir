// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
builtin.module {
    %0 = py.const 0
    %1 = py.const 1
    %2 = py.binop "add" %0 %1
    %3 = py.const "linalg.matmul"
    %4 = py.build_tuple()
    %5 = py.build_tuple(%3, %2, %4)
    %6 = py.const 1267650600228229401496703205376
    %7 = py.const -1267650600228229401496703205376
    %8 = py.const "" {test = "attribute"}
}

// CHECK:       builtin.module {
// CHECK-NEXT:      %0 = py.const 0
// CHECK-NEXT:      %1 = py.const 1
// CHECK-NEXT:      %2 = py.binop "add" %0 %1
// CHECK-NEXT:      %3 = py.const "linalg.matmul"
// CHECK-NEXT:      %4 = py.build_tuple()
// CHECK-NEXT:      %5 = py.build_tuple(%3, %2, %4)
// CHECK-NEXT:      %6 = py.const 1267650600228229401496703205376
// CHECK-NEXT:      %7 = py.const -1267650600228229401496703205376
// CHECK-NEXT:      %8 = py.const "" {test = "attribute"}
// CHECK-NEXT:  }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC:   %0 = "py.const"() <{const = #builtin.int<0>}> : () -> !py.object
// CHECK-GENERIC:   %1 = "py.const"() <{const = #builtin.int<1>}> : () -> !py.object
// CHECK-GENERIC:   %2 = "py.binop"(%0, %1) <{op = "add"}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC:   %3 = "py.const"() <{const = "linalg.matmul"}> : () -> !py.object
// CHECK-GENERIC:   %4 = "py.build_tuple"() : () -> !py.object
// CHECK-GENERIC:   %5 = "py.build_tuple"(%3, %2, %4) : (!py.object, !py.object, !py.object) -> !py.object
// CHECK-GENERIC:   %6 = "py.const"() <{const = #builtin.int<1267650600228229401496703205376>}> : () -> !py.object
// CHECK-GENERIC:   %7 = "py.const"() <{const = #builtin.int<-1267650600228229401496703205376>}> : () -> !py.object
// CHECK-GENERIC:   %8 = "py.const"() <{const = ""}> {test = "attribute"} : () -> !py.object
// CHECK-GENERIC: }) : () -> ()
