// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
    %0 = py.const 0
    %1 = py.const 1
    %2 = py.binop "add" %0 %1
    %3 = py.const "linalg.matmul"
    %4 = py.const 1267650600228229401496703205376
    %5 = py.const -1267650600228229401496703205376
    %6 = py.const "" {test = "attribute"}
    %7 = py.const true
    %8 = py.const false
    %9 = py.const 1.5
    %10 = py.const -2.0
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = py.const 0
// CHECK-NEXT:    %1 = py.const 1
// CHECK-NEXT:    %2 = py.binop "add" %0 %1
// CHECK-NEXT:    %3 = py.const "linalg.matmul"
// CHECK-NEXT:    %4 = py.const 1267650600228229401496703205376
// CHECK-NEXT:    %5 = py.const -1267650600228229401496703205376
// CHECK-NEXT:    %6 = py.const "" {test = "attribute"}
// CHECK-NEXT:    %7 = py.const true
// CHECK-NEXT:    %8 = py.const false
// CHECK-NEXT:    %9 = py.const 1.5
// CHECK-NEXT:    %10 = py.const -2.0
// CHECK-NEXT:  }

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %0 = "py.const"() <{const = #builtin.int<0>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %1 = "py.const"() <{const = #builtin.int<1>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %2 = "py.binop"(%0, %1) <{op = "add"}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %3 = "py.const"() <{const = "linalg.matmul"}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %4 = "py.const"() <{const = #builtin.int<1267650600228229401496703205376>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %5 = "py.const"() <{const = #builtin.int<-1267650600228229401496703205376>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %6 = "py.const"() <{const = ""}> {test = "attribute"} : () -> !py.object
// CHECK-GENERIC-NEXT:    %7 = "py.const"() <{const = true}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %8 = "py.const"() <{const = false}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %9 = "py.const"() <{const = 1.500000e+00 : f64}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %10 = "py.const"() <{const = -2.000000e+00 : f64}> : () -> !py.object
// CHECK-GENERIC-NEXT:  }) : () -> ()
