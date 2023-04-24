// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    "func.return"() : () -> ()
  }) {"sym_name" = "noarg_void", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

   // CHECK:      "func.func"() ({
   // CHECK-NEXT:   "func.return"() : () -> ()
   // CHECK-NEXT: }) {"sym_name" = "noarg_void", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
    "func.call"() {"callee" = @call_void} : () -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "call_void", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

   // CHECK: "func.func"() ({
   // CHECK-NEXT:   "func.call"() {"callee" = @call_void} : () -> ()
   // CHECK-NEXT:   "func.return"() : () -> ()
   // CHECK-NEXT: }) {"sym_name" = "call_void", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
  ^0(%0 : !test.type<"int">):
    %1 = "func.call"(%0) {"callee" = @arg_rec} : (!test.type<"int">) -> !test.type<"int">
    "func.return"(%1) : (!test.type<"int">) -> ()
  }) {"sym_name" = "arg_rec", "function_type" = (!test.type<"int">) -> !test.type<"int">, "sym_visibility" = "private"} : () -> ()

   // CHECK:      "func.func"() ({
   // CHECK-NEXT: ^0(%0 : !test.type<"int">):
   // CHECK-NEXT:   %1 = "func.call"(%0) {"callee" = @arg_rec} : (!test.type<"int">) -> !test.type<"int">
   // CHECK-NEXT:   "func.return"(%1) : (!test.type<"int">) -> ()
   // CHECK-NEXT: }) {"sym_name" = "arg_rec", "function_type" = (!test.type<"int">) -> !test.type<"int">, "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
  }) {"sym_name" = "external_fn", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()

   // CHECK:      "func.func"() ({
   // CHECK-NEXT: }) {"sym_name" = "external_fn", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()

}) : () -> ()


