// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

"builtin.module"() ({
  %lbi = "test.op"() {"value" = 0 : index} : () -> !test.type<"int">
  %ub = "test.op"() {"value" = 42 : index} : () -> index
  %step = "test.op"() {"value" = 7 : index} : () -> index
// CHECK: !test.type<"int"> should be of base attribute index
  "scf.for"(%lbi, %ub, %step) ({
  ^0(%iv : index):
    "scf.yield"() : () -> ()
  }) : (!test.type<"int">, index, index) -> ()
}) : () -> ()

// -----

"builtin.module"() ({
  %lb = "test.op"() {"value" = 0 : index} : () -> index
  %ub = "test.op"() {"value" = 42 : index} : () -> index
  %step = "test.op"() {"value" = 7 : index} : () -> index
  "scf.for"(%lb, %ub, %step) ({
  ^0(%iv : !test.type<"int">):
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
}) : () -> ()

// CHECK: The first block argument of the body is of type !test.type<"int"> instead of index
