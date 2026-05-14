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

// -----

// #acc.declare: `dataClause` is required (upstream
// `DefaultValuedParameter<>` only defaults `implicit`, not the
// clause). Omitting it trips the parser's missing-required check.
"test.op"() {x = #acc.declare<implicit = true>} : () -> ()
// CHECK: struct is missing required parameter: dataClause

// -----

// #acc.declare: each struct field may appear at most once. The
// parser rejects duplicates with the same diagnostic upstream uses.
"test.op"() {x = #acc.declare<dataClause = acc_create, dataClause = acc_copyin>} : () -> ()
// CHECK: duplicate struct parameter name: dataClause

// -----

// #acc.declare_action: each of the four slots may appear at most
// once. Repeating any one of them trips the parser's
// dedupe check (here exercised on `postAlloc`).
"test.op"() {x = #acc.declare_action<postAlloc = @a, postAlloc = @b>} : () -> ()
// CHECK: duplicate struct parameter name: postAlloc

// -----

// #acc.declare_action: every slot, if present, must hold a symbol
// reference. A string literal trips the parser's isinstance check
// with a per-slot diagnostic naming the offending field.
"test.op"() {x = #acc.declare_action<preAlloc = "foo">} : () -> ()
// CHECK: expected symbol reference for #acc.declare_action parameter 'preAlloc', got "foo"
