// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

"builtin.module"() ({
  %lbi = "test.op"() : () -> !test.type<"int">
  %x:2 = "test.op"() : () -> (index, index) // ub, step
// CHECK: operand at position 0 does not verify
  "scf.for"(%lbi, %x#0, %x#1) ({
  ^0(%iv : index):
    "scf.yield"() : () -> ()
  }) : (!test.type<"int">, index, index) -> ()
}) : () -> ()

// -----

"builtin.module"() ({
  %x:3 = "test.op"() : () -> (index, index, index) // lb, ub, step
  "scf.for"(%x#0, %x#1, %x#2) ({
  ^0(%iv : !test.type<"int">):
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()

// CHECK: Expected induction var to be same type as bounds and step

// -----

"builtin.module"() ({
  %lbi = "test.op"() : () -> si32
  %x:2 = "test.op"() : () -> (index, index) // ub, step
// CHECK: Operand #0 must be signless integer or index, but got si32
  "scf.for"(%lbi, %x#0, %x#1) ({
  ^0(%iv : index):
    "scf.yield"() : () -> ()
  }) : (si32, index, index) -> ()
}) : () -> ()

// -----

"builtin.module"() ({
  %x:3 = "test.op"() : () -> (index, index, index) // lb, ub, step
  "scf.for"(%x#0, %x#1, %x#2) ({
  ^0(%iv : i32):
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()

// CHECK: Expected induction var to be same type as bounds and step
