// XFAIL: *
// RUN: xdsl-opt -p convert-func-to-x86-func --split-input-file  %s | filecheck %s

func.func @foo_int() -> (i32,i32) {
  %a = "test.op"(): () -> i32
  %b = "test.op"(): () -> i32
  func.return %a,%b: i32,i32
}
