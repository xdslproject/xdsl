// RUN: xdsl-opt %s -p canonicalize | filecheck %s

%0 = "test.op"() : () -> !py.object<"int">
%1 = py.cast %0 : !py.object<"int"> to !py.object<"int">
%2 = py.cast %0 : !py.object<"int"> to !py.object
"test.op"(%1, %2) : (!py.object<"int">, !py.object) -> ()

// CHECK:      %0 = "test.op"() : () -> !py.object<"int">
// CHECK-NEXT: %1 = py.cast %0 : !py.object<"int"> to !py.object
// CHECK-NEXT: "test.op"(%0, %1) : (!py.object<"int">, !py.object) -> ()
