// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

// #acc.routine_info: every entry in the `[...]` payload must be a symbol
// reference (`@name`). A non-symref entry trips the parser's isinstance
// check; the diagnostic names the offending value so users can spot the
// bad entry in long lists.
"test.op"() {x = #acc.routine_info<["foo"]>} : () -> ()
// CHECK: expected symbol reference in #acc.routine_info, got "foo"

// -----

// #acc.specialized_routine: the first parameter (`$routine`) must be a
// symbol reference back to the originating `acc.routine`. Any other
// attribute kind is rejected at parse time.
"test.op"() {x = #acc.specialized_routine<"foo", <gang_dim1>, "bar">} : () -> ()
// CHECK: expected symbol reference in #acc.specialized_routine, got "foo"
