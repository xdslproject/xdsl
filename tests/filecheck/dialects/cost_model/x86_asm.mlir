// RUN: xdsl-opt -t riscv-asm --split-input-file --verify-diagnostics %s | filecheck %s


x86_func.func @noarg_void() {
  cost_model.begin_mca_region_of_interest
  cost_model.stop_mca_region_of_interest
  x86_func.ret
}

// CHECK:       noarg_void:
// CHECK-NEXT:      # LLVM-MCA-BEGIN
// CHECK-NEXT:      # LLVM-MCA-END
// CHECK-NEXT:      ret
