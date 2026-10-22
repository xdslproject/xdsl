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
    %9 = py.const true
    %10 = py.const false
    %11 = py.const 1.5
    %12 = py.const -2.0
    %13 = py.compare "lt" %0 %1
    %14 = py.bool %13
    %15 = py.call "f"(%0, %1)
    %16 = py.call "MyClass.method"(%0, %1, %2) keywords ["b", "c"]
    %17 = py.call "g"()
    %18, %19 = py.unpack %5 : !py.object, !py.object
    py.func @identity ["int"] -> "int" {
    ^bb0(%x : !py.object):
        py.return %x
    }
    py.func @nothing [] -> "None" {
        py.return
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = py.const 0
// CHECK-NEXT:    %1 = py.const 1
// CHECK-NEXT:    %2 = py.binop "add" %0 %1
// CHECK-NEXT:    %3 = py.const "linalg.matmul"
// CHECK-NEXT:    %4 = py.build_tuple()
// CHECK-NEXT:    %5 = py.build_tuple(%3, %2, %4)
// CHECK-NEXT:    %6 = py.const 1267650600228229401496703205376
// CHECK-NEXT:    %7 = py.const -1267650600228229401496703205376
// CHECK-NEXT:    %8 = py.const "" {test = "attribute"}
// CHECK-NEXT:    %9 = py.const true
// CHECK-NEXT:    %10 = py.const false
// CHECK-NEXT:    %11 = py.const 1.5
// CHECK-NEXT:    %12 = py.const -2.0
// CHECK-NEXT:    %13 = py.compare "lt" %0 %1
// CHECK-NEXT:    %14 = py.bool %13
// CHECK-NEXT:    %15 = py.call "f"(%0, %1)
// CHECK-NEXT:    %16 = py.call "MyClass.method"(%0, %1, %2) keywords ["b", "c"]
// CHECK-NEXT:    %17 = py.call "g"()
// CHECK-NEXT:    %18, %19 = py.unpack %5 : !py.object, !py.object
// CHECK-NEXT:    py.func @identity ["int"] -> "int" {
// CHECK-NEXT:    ^bb0(%x: !py.object):
// CHECK-NEXT:      py.return %x
// CHECK-NEXT:    }
// CHECK-NEXT:    py.func @nothing [] -> "None" {
// CHECK-NEXT:      py.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %0 = "py.const"() <{const = #builtin.int<0>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %1 = "py.const"() <{const = #builtin.int<1>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %2 = "py.binop"(%0, %1) <{op = "add"}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %3 = "py.const"() <{const = "linalg.matmul"}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %4 = "py.build_tuple"() : () -> !py.object
// CHECK-GENERIC-NEXT:    %5 = "py.build_tuple"(%3, %2, %4) : (!py.object, !py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %6 = "py.const"() <{const = #builtin.int<1267650600228229401496703205376>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %7 = "py.const"() <{const = #builtin.int<-1267650600228229401496703205376>}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %8 = "py.const"() <{const = ""}> {test = "attribute"} : () -> !py.object
// CHECK-GENERIC-NEXT:    %9 = "py.const"() <{const = true}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %10 = "py.const"() <{const = false}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %11 = "py.const"() <{const = 1.500000e+00 : f64}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %12 = "py.const"() <{const = -2.000000e+00 : f64}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %13 = "py.compare"(%0, %1) <{op = "lt"}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %14 = "py.bool"(%13) : (!py.object) -> i1
// CHECK-GENERIC-NEXT:    %15 = "py.call"(%0, %1) <{callee = "f"}> : (!py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %16 = "py.call"(%0, %1, %2) <{callee = "MyClass.method", keywords = ["b", "c"]}> : (!py.object, !py.object, !py.object) -> !py.object
// CHECK-GENERIC-NEXT:    %17 = "py.call"() <{callee = "g"}> : () -> !py.object
// CHECK-GENERIC-NEXT:    %18, %19 = "py.unpack"(%5) : (!py.object) -> (!py.object, !py.object)
// CHECK-GENERIC-NEXT:    "py.func"() <{sym_name = "identity", arg_annotations = ["int"], return_annotation = "int"}> ({
// CHECK-GENERIC-NEXT:    ^bb0(%x: !py.object):
// CHECK-GENERIC-NEXT:      "py.return"(%x) : (!py.object) -> ()
// CHECK-GENERIC-NEXT:    }) : () -> ()
// CHECK-GENERIC-NEXT:    "py.func"() <{sym_name = "nothing", arg_annotations = [], return_annotation = "None"}> ({
// CHECK-GENERIC-NEXT:      "py.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()
