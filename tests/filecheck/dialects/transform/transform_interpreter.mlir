// RUN: xdsl-opt %s -p transform-interpreter | filecheck %s


module {

  "builtin.module"() ({
    "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
    ^bb0(%arg0: !transform.any_op):
      "transform.yield"() : () -> ()
    }) : () -> ()
  }) {transform.with_named_sequence} : () -> ()
}

// CHECK: "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
// CHECK-NEXT: ^0(%arg0 : !transform.any_op):
// CHECK-NEXT:   transform.yield
// CHECK-NEXT: }) : () -> ()
