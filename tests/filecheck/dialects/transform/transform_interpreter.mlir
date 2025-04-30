// RUN: xdsl-opt %s -p transform-interpreter | filecheck %s
// RUN: xdsl-opt %s -p transform-interpreter'{entry-point=entry}' | filecheck %s


module {

  module attributes {transform.with_named_sequence} {
    "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
    ^bb0(%arg0: !transform.any_op):
      transform.yield
    }) : () -> ()
  }

  module attributes {transform.with_named_sequence} {
    "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "entry"}> ({
    ^bb0(%arg0: !transform.any_op):
      transform.yield
    }) : () -> ()
  }
}

// CHECK: "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
// CHECK-NEXT: ^0(%arg0 : !transform.any_op):
// CHECK-NEXT:   transform.yield
// CHECK-NEXT: }) : () -> ()

// CHECK: "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "entry"}> ({
// CHECK-NEXT: ^0(%arg0 : !transform.any_op):
// CHECK-NEXT:   transform.yield
// CHECK-NEXT: }) : () -> ()
