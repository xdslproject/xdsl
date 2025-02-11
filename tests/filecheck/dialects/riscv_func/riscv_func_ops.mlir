// RUN: xdsl-opt --print-op-generic %s | xdsl-opt | filecheck %s

builtin.module {

  riscv_func.func @noarg_void() {
    riscv_func.return
  }

   // CHECK:      riscv_func.func @noarg_void() {
   // CHECK-NEXT:   riscv_func.return
   // CHECK-NEXT: }

  riscv_func.func @call_void() {
    riscv_func.call @call_void() : () -> ()
    riscv_func.return
  }

   // CHECK: riscv_func.func @call_void() {
   // CHECK-NEXT:   riscv_func.call @call_void() : () -> ()
   // CHECK-NEXT:   riscv_func.return
   // CHECK-NEXT: }

  riscv_func.func @call_void_attributes() {
    riscv_func.call @call_void_attributes() {"hello" = "world"} : () -> ()
    riscv_func.return
  }

   // CHECK: riscv_func.func @call_void_attributes() {
   // CHECK-NEXT:   riscv_func.call @call_void_attributes() {hello = "world"} : () -> ()
   // CHECK-NEXT:   riscv_func.return
   // CHECK-NEXT: }

  riscv_func.func @arg_rec(%0 : !riscv.reg) -> !riscv.reg {
    %1 = riscv_func.call @arg_rec(%0) : (!riscv.reg) -> !riscv.reg
    riscv_func.return %1 : !riscv.reg
  }

   // CHECK: riscv_func.func @arg_rec(%0 : !riscv.reg) -> !riscv.reg {
   // CHECK-NEXT:   %{{.*}} = riscv_func.call @arg_rec(%{{.*}}) : (!riscv.reg) -> !riscv.reg
   // CHECK-NEXT:   riscv_func.return %{{.*}} : !riscv.reg
   // CHECK-NEXT: }

  riscv_func.func @arg_rec_block(!riscv.reg) -> !riscv.reg {
  ^0(%2 : !riscv.reg):
    %3 = riscv_func.call @arg_rec_block(%2) : (!riscv.reg) -> !riscv.reg
    riscv_func.return %3 : !riscv.reg
  }

  // CHECK: riscv_func.func @arg_rec_block(%{{\d+}} : !riscv.reg) -> !riscv.reg {
  // CHECK-NEXT:   %{{\d+}} = riscv_func.call @arg_rec_block(%{{\d+}}) : (!riscv.reg) -> !riscv.reg
  // CHECK-NEXT:   riscv_func.return %{{\d+}} : !riscv.reg
  // CHECK-NEXT: }

  riscv_func.func @multi_return_body(%a : !riscv.reg) -> (!riscv.reg, !riscv.reg) {
    riscv_func.return %a, %a : !riscv.reg, !riscv.reg
  }

  // CHECK: riscv_func.func @multi_return_body(%a : !riscv.reg) -> (!riscv.reg, !riscv.reg) {
  // CHECK-NEXT:   riscv_func.return %a, %a : !riscv.reg, !riscv.reg
  // CHECK-NEXT: }

  riscv_func.func private @visibility_private() {
    riscv_func.return
  }

  // CHECK: riscv_func.func private @visibility_private() {
  // CHECK-NEXT:   riscv_func.return
  // CHECK-NEXT: }

  riscv_func.func public @visibility_public() {
    riscv_func.return
  }

  // CHECK: riscv_func.func public @visibility_public() {
  // CHECK-NEXT:   riscv_func.return
  // CHECK-NEXT: }

}
