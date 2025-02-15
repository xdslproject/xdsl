// RUN: XDSL_ROUNDTRIP

x86.assembly_section ".text" {
  x86.directive ".global" "external"
  x86_func.func @external() -> ()
}

// CHECK:      builtin.module {
// CHECK-NEXT:   x86.assembly_section ".text" {
// CHECK-NEXT:     x86.directive ".global" "external"
// CHECK-NEXT:     x86_func.func @external() -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
