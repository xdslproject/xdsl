// XFAIL: *
// RUN: xdsl-opt -p convert-func-to-x86-func --split-input-file  %s | filecheck %s

func.func @f() -> (memref<1xf32>) {
  %a = memref.alloc(): memref<1xf32>
  func.return %a: memref<1xf32>
}
