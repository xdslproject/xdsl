// RUN: xdsl-opt -p transform-interpreter --split-input-file --verify-diagnostics %s | filecheck %s
// RUN: xdsl-opt -p 'transform-interpreter{entry-point=entry}' --split-input-file --verify-diagnostics %s | filecheck %s --check-prefix=ENTRY

module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
      printf.print_format "executed default entry\n"
      transform.yield
    }
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @entry(%root: !transform.any_op {transform.readonly}) {
      printf.print_format "executed custom entry\n"
      transform.yield
    }
  }
}


// The string will be printed before the rest of the IR.

//  CHECK-NOT:  {{^}}executed custom entry{{$}}
//      CHECK:  executed default entry
// CHECK-NEXT:  builtin.module

//  ENTRY-NOT:  {{^}}executed default entry{{$}}
//      ENTRY:  executed custom entry
// ENTRY-NEXT:  builtin.module

// -----

module {}

// A missing entry point reports the requested name.

// CHECK: could not find a nested named sequence with name: __transform_main
// ENTRY: could not find a nested named sequence with name: entry
