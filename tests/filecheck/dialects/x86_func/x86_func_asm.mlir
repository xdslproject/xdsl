// RUN: xdsl-opt -t riscv-asm --split-input-file --verify-diagnostics %s | filecheck %s


x86_func.func @noarg_void() {
  x86_func.ret
}

// CHECK:       noarg_void:
// CHECK-NEXT:      ret


x86_func.func private @visibility_private() {
  x86_func.ret
}

// CHECK-NEXT:  .local visibility_private
// CHECK-NEXT:  visibility_private:
// CHECK-NEXT:      ret

x86_func.func public @visibility_public() {
  x86_func.ret
}

// CHECK-NEXT:  .globl visibility_public
// CHECK-NEXT:  visibility_public:
// CHECK-NEXT:      ret


// -----

"x86_func.func"() ({
"x86_func.ret"() : () -> ()
}) {sym_name = "visibility_invalid", function_type = () -> (), sym_visibility = "invalid_visibility"} : () -> ()

// CHECK:  Unexpected visibility invalid_visibility for function "visibility_invalid"
