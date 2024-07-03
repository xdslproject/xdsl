// RUN: xdsl-opt -p convert-print-format-to-riscv-debug %s | filecheck %s

%i32, %i64, %f32, %f64 = "test.op"() : () -> (i32, i64, f32, f64)

printf.print_format "Hello world!"
printf.print_format "bitwidths {} {} {} {}", %i32 : i32, %i64 : i64, %f32 : f32, %f64 : f64

// CHECK:      builtin.module {
// CHECK-NEXT:   %i32, %i64, %f32, %f64 = "test.op"() : () -> (i32, i64, f32, f64)
// CHECK-NEXT:   riscv_debug.printf "Hello world!" : () -> ()
// CHECK-NEXT:   %0 = builtin.unrealized_conversion_cast %i32 : i32 to !riscv.reg
// CHECK-NEXT:   %1 = builtin.unrealized_conversion_cast %i64 : i64 to !riscv.reg
// CHECK-NEXT:   %2 = builtin.unrealized_conversion_cast %f32 : f32 to !riscv.freg
// CHECK-NEXT:   %3 = builtin.unrealized_conversion_cast %f64 : f64 to !riscv.freg
// CHECK-NEXT:   riscv_debug.printf %0, %1, %2, %3 "bitwidths {} {} {} {}" : (!riscv.reg, !riscv.reg, !riscv.freg, !riscv.freg) -> ()
// CHECK-NEXT: }
