// RUN: xdsl-opt %s -t wat | filecheck %s

wasm.module

// CHECK: (module)
