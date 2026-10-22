// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%init = "test.op"() : () -> i32
%res = "scf.while"(%init) ({
^bb0(%arg : i32):
  %cond = "test.op"() : () -> i1
  "scf.condition"(%cond, %arg) : (i1, i32) -> ()
}, {
^bb1(%body_arg : i64):
  "scf.yield"() : () -> ()
}) : (i32) -> i32

// CHECK: 'scf.while': along control flow edge from Operation 'scf.condition' to Region #1: successor operand type #0 i32 should match successor input type #0 i64

// -----

%init = "test.op"() : () -> i32
%res = "scf.while"(%init) ({
^bb0(%arg : i32):
  %cond = "test.op"() : () -> i1
  "scf.condition"(%cond, %arg) : (i1, i32) -> ()
}, {
^bb1(%body_arg : i32):
  %next = "test.op"() : () -> i64
  "scf.yield"(%next) : (i64) -> ()
}) : (i32) -> i32

// CHECK: 'scf.while': along control flow edge from Operation 'scf.yield' to Region #0: successor operand type #0 i64 should match successor input type #0 i32

// -----

%cond = "test.op"() : () -> i1
%res = "scf.if"(%cond) ({
  %a = "test.op"() : () -> i32
  "scf.yield"(%a) : (i32) -> ()
}, {
}) : (i1) -> i32

// CHECK: 'scf.if': along control flow edge from parent to Operation 'scf.if': region branch point has 0 operands, but region successor needs 1 inputs

// -----

%res = "scf.execute_region"() ({
  %a = "test.op"() : () -> i64
  "scf.yield"(%a) : (i64) -> ()
}) : () -> i32

// CHECK: 'scf.execute_region': along control flow edge from Operation 'scf.yield' to Operation 'scf.execute_region': successor operand type #0 i64 should match successor input type #0 i32
