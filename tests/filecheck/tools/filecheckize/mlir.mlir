// RUN: filecheckize %s --strip-comments | filecheck %s --match-full-lines

func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
    %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
    func.return %1 : !test.type<"int">
}

// CHECK:      // CHECK:      func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
// CHECK-NEXT: // CHECK-NEXT:     %1 = func.call @arg_rec(%0) : (!test.type<"int">) -> !test.type<"int">
// CHECK-NEXT: // CHECK-NEXT:     func.return %1 : !test.type<"int">
// CHECK-NEXT: // CHECK-NEXT: }