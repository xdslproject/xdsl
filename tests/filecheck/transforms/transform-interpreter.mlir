// RUN: xdsl-opt %s -p transform-interpreter --split-input-file --verify-diagnostics | filecheck %s
// RUN: xdsl-opt %s -p 'transform-interpreter{entry-point=entry}' --split-input-file --verify-diagnostics | filecheck %s --check-prefix=ENTRY

// Find and execute only the selected entry point inside a nested module. Match
// complete printed lines so the printf operations in the output IR cannot satisfy
// the checks. Exactly one marker must appear before the output module.
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

// CHECK-NOT: {{^}}executed custom entry{{$}}
// CHECK: {{^}}executed default entry{{$}}

// ENTRY-NOT: {{^}}executed default entry{{$}}
// ENTRY: {{^}}executed custom entry{{$}}

// -----

// A missing entry point reports the requested name. --verify-diagnostics prints
// the pass failure so FileCheck can check it without matching a Python traceback.
module {}

// CHECK: could not find a nested named sequence with name: __transform_main
// ENTRY: could not find a nested named sequence with name: entry
