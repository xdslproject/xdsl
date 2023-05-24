// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s

builtin.module {

  func.func @cus_noarg_void() {
    "func.return"() : () -> ()
  }

   // CHECK:      "func.func"() ({
   // CHECK-NEXT:   "func.return"() : () -> ()
   // CHECK-NEXT: }) {"sym_name" = "cus_noarg_void", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

  func.func @call_void() {
    "func.call"() {"callee" = @call_void} : () -> ()
    "func.return"() : () -> ()
  }

   // CHECK-NEXT: "func.func"() ({
   // CHECK-NEXT:   "func.call"() {"callee" = @call_void} : () -> ()
   // CHECK-NEXT:   "func.return"() : () -> ()
   // CHECK-NEXT: }) {"sym_name" = "call_void", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

  func.func @arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
    %1 = "func.call"(%0) {"callee" = @arg_rec} : (!test.type<"int">) -> !test.type<"int">
    "func.return"(%1) : (!test.type<"int">) -> ()
  }

   // CHECK-NEXT:      "func.func"() ({
   // CHECK-NEXT: ^{{.*}}(%{{.*}} : !test.type<"int">):
   // CHECK-NEXT:   %{{.*}} = "func.call"(%{{.*}}) {"callee" = @arg_rec} : (!test.type<"int">) -> !test.type<"int">
   // CHECK-NEXT:   "func.return"(%{{.*}}) : (!test.type<"int">) -> ()
   // CHECK-NEXT: }) {"sym_name" = "arg_rec", "function_type" = (!test.type<"int">) -> !test.type<"int">, "sym_visibility" = "private"} : () -> ()

  func.func @arg_rec_block(!test.type<"int">) -> !test.type<"int"> {
  ^0(%0 : !test.type<"int">):
    %1 = "func.call"(%0) {"callee" = @arg_rec_block} : (!test.type<"int">) -> !test.type<"int">
    "func.return"(%1) : (!test.type<"int">) -> ()
  }

   // CHECK-NEXT:      "func.func"() ({
   // CHECK-NEXT: ^{{.*}}(%{{.*}} : !test.type<"int">):
   // CHECK-NEXT:   %{{.*}} = "func.call"(%{{.*}}) {"callee" = @arg_rec_block} : (!test.type<"int">) -> !test.type<"int">
   // CHECK-NEXT:   "func.return"(%{{.*}}) : (!test.type<"int">) -> ()
   // CHECK-NEXT: }) {"sym_name" = "arg_rec_block", "function_type" = (!test.type<"int">) -> !test.type<"int">, "sym_visibility" = "private"} : () -> ()

  func.func @external_fn(i32) -> (i32, i32)

   // CHECK-NEXT:      "func.func"() ({
   // CHECK-NEXT: }) {"sym_name" = "external_fn", "function_type" = (i32) -> (i32, i32), "sym_visibility" = "private"} : () -> ()
}
