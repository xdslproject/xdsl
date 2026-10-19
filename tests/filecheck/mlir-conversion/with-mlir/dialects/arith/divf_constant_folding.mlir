// RUN: xdsl-opt "%s" -p canonicalize | filecheck "%s"
// RUN: $XDSL_MLIR_OPT --canonicalize "%s" | xdsl-opt | filecheck "%s"

// No fast-math flags: division must preserve zero signs and propagate NaNs.
// The final xdsl-opt in the MLIR run only normalizes printing.
// Separate functions keep each case independent of constant deduplication.

func.func @positive_by_positive_zero() -> f64 {
  %a = arith.constant 1.0 : f64
  %b = arith.constant 0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @positive_by_positive_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff0000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @positive_by_negative_zero() -> f64 {
  %a = arith.constant 1.0 : f64
  %b = arith.constant -0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @positive_by_negative_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0xfff0000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @negative_by_positive_zero() -> f64 {
  %a = arith.constant -1.0 : f64
  %b = arith.constant 0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @negative_by_positive_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0xfff0000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @negative_by_negative_zero() -> f64 {
  %a = arith.constant -1.0 : f64
  %b = arith.constant -0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @negative_by_negative_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff0000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @infinity_by_positive_zero() -> f64 {
  %a = arith.constant 0x7ff0000000000000 : f64
  %b = arith.constant 0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @infinity_by_positive_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff0000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @infinity_by_negative_zero() -> f64 {
  %a = arith.constant 0x7ff0000000000000 : f64
  %b = arith.constant -0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @infinity_by_negative_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0xfff0000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @negative_infinity_by_positive_zero() -> f64 {
  %a = arith.constant 0xfff0000000000000 : f64
  %b = arith.constant 0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @negative_infinity_by_positive_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0xfff0000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @negative_infinity_by_negative_zero() -> f64 {
  %a = arith.constant 0xfff0000000000000 : f64
  %b = arith.constant -0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @negative_infinity_by_negative_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff0000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @nan_by_positive_zero() -> f64 {
  %a = arith.constant 0x7ff8000000000000 : f64
  %b = arith.constant 0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @nan_by_positive_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff8000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @nan_by_negative_zero() -> f64 {
  %a = arith.constant 0x7ff8000000000000 : f64
  %b = arith.constant -0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @nan_by_negative_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff8000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @positive_zero_by_positive_zero() -> f64 {
  %a = arith.constant 0.0 : f64
  %r = arith.divf %a, %a : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @positive_zero_by_positive_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff8000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @positive_zero_by_negative_zero() -> f64 {
  %a = arith.constant 0.0 : f64
  %b = arith.constant -0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @positive_zero_by_negative_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff8000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @negative_zero_by_positive_zero() -> f64 {
  %a = arith.constant -0.0 : f64
  %b = arith.constant 0.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @negative_zero_by_positive_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff8000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @negative_zero_by_negative_zero() -> f64 {
  %a = arith.constant -0.0 : f64
  %r = arith.divf %a, %a : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @negative_zero_by_negative_zero
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7ff8000000000000 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @positive_zero_by_negative() -> f64 {
  %a = arith.constant 0.0 : f64
  %b = arith.constant -1.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @positive_zero_by_negative
// CHECK-NEXT: %[[R:.*]] = arith.constant -0.000000e+00 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @negative_zero_by_negative() -> f64 {
  %a = arith.constant -0.0 : f64
  %b = arith.constant -1.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @negative_zero_by_negative
// CHECK-NEXT: %[[R:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @finite_division() -> f64 {
  %a = arith.constant 6.0 : f64
  %b = arith.constant 2.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @finite_division
// CHECK-NEXT: %[[R:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @finite_negative_division() -> f64 {
  %a = arith.constant 6.0 : f64
  %b = arith.constant -2.0 : f64
  %r = arith.divf %a, %b : f64
  func.return %r : f64
}
// CHECK-LABEL: func.func @finite_negative_division
// CHECK-NEXT: %[[R:.*]] = arith.constant -3.000000e+00 : f64
// CHECK-NEXT: func.return %[[R]] : f64

func.func @negative_zero_divisor_f16() -> f16 {
  %a = arith.constant 1.0 : f16
  %b = arith.constant -0.0 : f16
  %r = arith.divf %a, %b : f16
  func.return %r : f16
}
// CHECK-LABEL: func.func @negative_zero_divisor_f16
// CHECK-NEXT: %[[R:.*]] = arith.constant 0xfc00 : f16
// CHECK-NEXT: func.return %[[R]] : f16

func.func @nan_zero_divisor_f16() -> f16 {
  %a = arith.constant 0x7e00 : f16
  %b = arith.constant 0.0 : f16
  %r = arith.divf %a, %b : f16
  func.return %r : f16
}
// CHECK-LABEL: func.func @nan_zero_divisor_f16
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7e00 : f16
// CHECK-NEXT: func.return %[[R]] : f16

func.func @negative_zero_divisor_f32() -> f32 {
  %a = arith.constant 1.0 : f32
  %b = arith.constant -0.0 : f32
  %r = arith.divf %a, %b : f32
  func.return %r : f32
}
// CHECK-LABEL: func.func @negative_zero_divisor_f32
// CHECK-NEXT: %[[R:.*]] = arith.constant 0xff800000 : f32
// CHECK-NEXT: func.return %[[R]] : f32

func.func @nan_zero_divisor_f32() -> f32 {
  %a = arith.constant 0x7fc00000 : f32
  %b = arith.constant 0.0 : f32
  %r = arith.divf %a, %b : f32
  func.return %r : f32
}
// CHECK-LABEL: func.func @nan_zero_divisor_f32
// CHECK-NEXT: %[[R:.*]] = arith.constant 0x7fc00000 : f32
// CHECK-NEXT: func.return %[[R]] : f32
