// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
    %0 = py.const 0
    %1 = py.const 1
    %2 = py.binop add, %0, %1
    %3 = py.const "linalg.matmul"
    %4 = py.const 1267650600228229401496703205376
    %5 = py.const -1267650600228229401496703205376
    %6 = py.const "" {test = "attribute"}
    %7 = py.const true
    %8 = py.const false
    %9 = py.const 1.5
    %10 = py.const -2.0
    %11 = py.call "f"(%0, %1)
    %12 = py.call "MyClass.method"(%0, %1, %2) keywords ["b", "c"]
    %13 = py.call "g"()
    %14 = py.compare lt, %0, %1
    %15 = py.compare notin, %0, %1
    %16 = py.build_tuple()
    %17 = py.build_tuple(%3, %2, %16)
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = py.const 0
// CHECK-NEXT:    %1 = py.const 1
// CHECK-NEXT:    %2 = py.binop add, %0, %1
// CHECK-NEXT:    %3 = py.const "linalg.matmul"
// CHECK-NEXT:    %4 = py.const 1267650600228229401496703205376
// CHECK-NEXT:    %5 = py.const -1267650600228229401496703205376
// CHECK-NEXT:    %6 = py.const "" {test = "attribute"}
// CHECK-NEXT:    %7 = py.const true
// CHECK-NEXT:    %8 = py.const false
// CHECK-NEXT:    %9 = py.const 1.5
// CHECK-NEXT:    %10 = py.const -2.0
// CHECK-NEXT:    %11 = py.call "f"(%0, %1)
// CHECK-NEXT:    %12 = py.call "MyClass.method"(%0, %1, %2) keywords ["b", "c"]
// CHECK-NEXT:    %13 = py.call "g"()
// CHECK-NEXT:    %14 = py.compare lt, %0, %1
// CHECK-NEXT:    %15 = py.compare notin, %0, %1
// CHECK-NEXT:    %16 = py.build_tuple()
// CHECK-NEXT:    %17 = py.build_tuple(%3, %2, %16)
// CHECK-NEXT:  }

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %0 = "py.const"() <{const = #builtin.int<0>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %1 = "py.const"() <{const = #builtin.int<1>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %2 = "py.binop"(%0, %1) <{op = #py<operator add>}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %3 = "py.const"() <{const = "linalg.matmul"}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %4 = "py.const"() <{const = #builtin.int<1267650600228229401496703205376>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %5 = "py.const"() <{const = #builtin.int<-1267650600228229401496703205376>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %6 = "py.const"() <{const = ""}> {test = "attribute"} : () -> !py.object
// CHECK-GENERIC-NEXT:    %7 = "py.const"() <{const = true}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %8 = "py.const"() <{const = false}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %9 = "py.const"() <{const = 1.500000e+00 : f64}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %10 = "py.const"() <{const = -2.000000e+00 : f64}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %11 = "py.call"(%0, %1) <{callee = "f"}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %12 = "py.call"(%0, %1, %2) <{callee = "MyClass.method", keywords = ["b", "c"]}> : (!py.object, !py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %13 = "py.call"() <{callee = "g"}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %14 = "py.compare"(%0, %1) <{op = #py<cmpop lt>}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %15 = "py.compare"(%0, %1) <{op = #py<cmpop notin>}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %16 = "py.build_tuple"() : () -> !py.object
// CHECK-GENERIC-NEXT:    %17 = "py.build_tuple"(%3, %2, %16) : (!py.object, !py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:  }) : () -> ()
