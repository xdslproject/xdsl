// RUN: MLIR_GENERIC_ROUNDTRIP

builtin.module {
  func.func @acc_parallel_empty() {
    "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  func.func @acc_parallel_default_and_unit_attrs() {
    "acc.parallel"() <{defaultAttr = #acc<defaultvalue present>, selfAttr, combined, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  func.func @acc_parallel_device_type_arrays() {
    "acc.parallel"() <{asyncOnly = [#acc.device_type<nvidia>], waitOnly = [#acc.device_type<host>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  func.func @acc_serial_empty() {
    "acc.serial"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  func.func @acc_serial_default_and_unit_attrs() {
    "acc.serial"() <{defaultAttr = #acc<defaultvalue present>, selfAttr, combined, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  func.func @acc_serial_device_type_arrays() {
    "acc.serial"() <{asyncOnly = [#acc.device_type<nvidia>], waitOnly = [#acc.device_type<host>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @acc_parallel_empty() {
// CHECK-NEXT:      "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @acc_parallel_default_and_unit_attrs() {
// CHECK-NEXT:      "acc.parallel"() <{combined, defaultAttr = #acc<defaultvalue present>, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, selfAttr}> ({
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @acc_parallel_device_type_arrays() {
// CHECK-NEXT:      "acc.parallel"() <{asyncOnly = [#acc.device_type<nvidia>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, waitOnly = [#acc.device_type<host>]}> ({
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
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
// CHECK-NEXT:    func.func @acc_serial_device_type_arrays() {
// CHECK-NEXT:      acc.serial {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      } attributes {asyncOnly = [#acc.device_type<nvidia>], waitOnly = [#acc.device_type<host>]}
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
