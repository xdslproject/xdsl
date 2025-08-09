// RUN: xdsl-opt -t x86-asm --split-input-file --verify-diagnostics %s | filecheck %s


riscv_func.func @noarg_void() {
  riscv_func.return
}

// CHECK:       noarg_void:
// CHECK-NEXT:      ret

riscv_func.func @call_void() {
  riscv_func.call @call_void() : () -> ()
  riscv_func.return
}

// CHECK-NEXT:  call_void:
// CHECK-NEXT:      jal call_void
// CHECK-NEXT:      ret

riscv_func.func @call_void_attributes() {
  riscv_func.call @call_void_attributes() {"hello" = "world"} : () -> ()
  riscv_func.return
}

// CHECK-NEXT:  call_void_attributes:
// CHECK-NEXT:      jal call_void_attributes
// CHECK-NEXT:      ret

riscv_func.func @arg_rec(%0 : !riscv.reg) -> !riscv.reg {
  %1 = riscv_func.call @arg_rec(%0) : (!riscv.reg) -> !riscv.reg
  riscv_func.return %1 : !riscv.reg
}

// CHECK-NEXT:  arg_rec:
// CHECK-NEXT:      jal arg_rec
// CHECK-NEXT:      ret

riscv_func.func @arg_rec_block(!riscv.reg) -> !riscv.reg {
^bb0(%2 : !riscv.reg):
  %3 = riscv_func.call @arg_rec_block(%2) : (!riscv.reg) -> !riscv.reg
  riscv_func.return %3 : !riscv.reg
}

// CHECK-NEXT:  arg_rec_block:
// CHECK-NEXT:      jal arg_rec_block
// CHECK-NEXT:      ret

riscv_func.func @multi_return_body(%a : !riscv.reg) -> (!riscv.reg, !riscv.reg) {
  riscv_func.return %a, %a : !riscv.reg, !riscv.reg
}

// CHECK-NEXT:  multi_return_body:
// CHECK-NEXT:      ret

riscv_func.func private @visibility_private() {
  riscv_func.return
}

// CHECK-NEXT:  .local visibility_private
// CHECK-NEXT:  visibility_private:
// CHECK-NEXT:      ret

riscv_func.func public @visibility_public() {
  riscv_func.return
}

// CHECK-NEXT:  .globl visibility_public
// CHECK-NEXT:  visibility_public:
// CHECK-NEXT:      ret


// -----

"riscv_func.func"() ({
"riscv_func.return"() : () -> ()
}) {sym_name = "visibility_invalid", function_type = () -> (), sym_visibility = "invalid_visibility"} : () -> ()

// CHECK:  Unexpected visibility invalid_visibility for function "visibility_invalid"
