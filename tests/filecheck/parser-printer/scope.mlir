// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

module {
  func.func public @my_func() {
    return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func public @my_func() {
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "func.func"() <{sym_name = "my_func", function_type = () -> (), sym_visibility = "public"}> ({
// CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
