// RUN: MLIR_ROUNDTRIP

builtin.module {
  func.func @acc_serial_empty() {
    acc.serial {
      acc.yield
    }
    func.return
  }
  func.func @acc_serial_default_and_unit_attrs() {
    acc.serial combined(loop) {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  func.func @acc_serial_if_self_operands(%cond: i1) {
    acc.serial self(%cond) if(%cond) {
      acc.yield
    }
    func.return
  }
  func.func @acc_serial_async_operand(%v: i32) {
    acc.serial async(%v : i32) {
      acc.yield
    }
    func.return
  }
  func.func @acc_serial_wait_operands(%w: i64, %x: index) {
    acc.serial wait({%w : i64, %x : index}) {
      acc.yield
    }
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @acc_serial_empty() {
// CHECK-NEXT:      acc.serial {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @acc_serial_default_and_unit_attrs() {
// CHECK-NEXT:      acc.serial combined(loop) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @acc_serial_if_self_operands(%{{[^ :]+}}: i1) {
// CHECK-NEXT:      acc.serial self(%{{[^)]+}}) if(%{{[^)]+}}) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @acc_serial_async_operand(%{{[^ :]+}}: i32) {
// CHECK-NEXT:      acc.serial async(%{{[^ ]+}} : i32) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @acc_serial_wait_operands(%{{[^ :]+}}: i64, %{{[^ :]+}}: index) {
// CHECK-NEXT:      acc.serial wait({%{{[^ ]+}} : i64, %{{[^ ]+}} : index}) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
