// RUN: xdsl-opt %s -p convert-pdl-to-pdl-interp{convert_individually=true} | filecheck %s

// CHECK-LABEL: module @empty_module
module @empty_module {
// CHECK: func @matcher(%{{.*}}: !pdl.operation)
// CHECK-NEXT: pdl_interp.finalize
}
