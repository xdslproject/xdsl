// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func @acc_parallel_empty() {
    "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  // CHECK:      "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  // CHECK-NEXT:   acc.yield
  // CHECK-NEXT: }) : () -> ()

  func.func @acc_parallel_yield_operands(%arg0: memref<10xf32>) {
    "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"(%arg0) : (memref<10xf32>) -> ()
    }) : () -> ()
    func.return
  }
  // CHECK:      "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  // CHECK-NEXT:   acc.yield %{{.*}} : memref<10xf32>
  // CHECK-NEXT: }) : () -> ()

  func.func @acc_parallel_default_and_unit_attrs() {
    "acc.parallel"() <{defaultAttr = #acc<defaultvalue present>, selfAttr, combined, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  // CHECK:      "acc.parallel"() <{defaultAttr = #acc<defaultvalue present>, selfAttr, combined, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  // CHECK-NEXT:   acc.yield
  // CHECK-NEXT: }) : () -> ()

  func.func @acc_parallel_default_none() {
    "acc.parallel"() <{defaultAttr = #acc<defaultvalue none>, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  // CHECK: "acc.parallel"() <{defaultAttr = #acc<defaultvalue none>

  func.func @acc_parallel_device_type_arrays() {
    "acc.parallel"() <{asyncOnly = [#acc.device_type<nvidia>, #acc.device_type<host>], asyncOperandsDeviceType = [#acc.device_type<nvidia>], waitOnly = [#acc.device_type<multicore>], waitOperandsDeviceType = [#acc.device_type<default>], waitOperandsSegments = array<i32: 1>, hasWaitDevnum = [false], numGangsDeviceType = [#acc.device_type<star>], numGangsSegments = array<i32: 0>, numWorkersDeviceType = [#acc.device_type<radeon>], vectorLengthDeviceType = [#acc.device_type<none>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  // CHECK:      "acc.parallel"() <{
  // CHECK-SAME: asyncOnly = [#acc.device_type<nvidia>, #acc.device_type<host>]
  // CHECK-SAME: asyncOperandsDeviceType = [#acc.device_type<nvidia>]
  // CHECK-SAME: waitOnly = [#acc.device_type<multicore>]
  // CHECK-SAME: waitOperandsDeviceType = [#acc.device_type<default>]
  // CHECK-SAME: waitOperandsSegments = array<i32: 1>
  // CHECK-SAME: hasWaitDevnum = [false]
  // CHECK-SAME: numGangsDeviceType = [#acc.device_type<star>]
  // CHECK-SAME: numGangsSegments = array<i32: 0>
  // CHECK-SAME: numWorkersDeviceType = [#acc.device_type<radeon>]
  // CHECK-SAME: vectorLengthDeviceType = [#acc.device_type<none>]

  // Generic-form sanity roundtrip for acc.serial. Keeps us honest that the
  // generic emission stays valid even with the custom assembly format enabled.
  func.func @acc_serial_generic_empty() {
    "acc.serial"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  // CHECK:      acc.serial {
  // CHECK-NEXT:   acc.yield
  // CHECK-NEXT: }

  func.func @acc_serial_empty() {
    acc.serial {
      acc.yield
    }
    func.return
  }
  // CHECK:      acc.serial {
  // CHECK-NEXT:   acc.yield
  // CHECK-NEXT: }

  func.func @acc_serial_yield_operands(%arg0: memref<10xf32>) {
    acc.serial {
      acc.yield %arg0 : memref<10xf32>
    }
    func.return
  }
  // CHECK:      acc.serial {
  // CHECK-NEXT:   acc.yield %{{.*}} : memref<10xf32>
  // CHECK-NEXT: }

  func.func @acc_serial_async_operand(%v: i32) {
    acc.serial async(%v : i32) {
      acc.yield
    }
    func.return
  }
  // CHECK: acc.serial async(%{{.*}} : i32) {

  func.func @acc_serial_async_only() {
    acc.serial {
      acc.yield
    } attributes {asyncOnly = [#acc.device_type<none>]}
    func.return
  }
  // CHECK: acc.serial {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } attributes {asyncOnly = [#acc.device_type<none>]}

  func.func @acc_serial_wait_operand(%v: i32) {
    acc.serial wait(%v : i32) {
      acc.yield
    }
    func.return
  }
  // CHECK: acc.serial wait(%{{.*}} : i32) {

  func.func @acc_serial_wait_operands_multi(%v: i64, %w: i32, %x: index) {
    acc.serial wait(%v, %w, %x : i64, i32, index) {
      acc.yield
    }
    func.return
  }
  // CHECK: acc.serial wait(%{{.*}}, %{{.*}}, %{{.*}} : i64, i32, index) {

  func.func @acc_serial_wait_only() {
    acc.serial {
      acc.yield
    } attributes {waitOnly = [#acc.device_type<none>]}
    func.return
  }
  // CHECK: acc.serial {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } attributes {waitOnly = [#acc.device_type<none>]}

  func.func @acc_serial_private_firstprivate(%a: memref<10xf32>, %b: memref<10x10xf32>) {
    acc.serial firstprivate(%a : memref<10xf32>) private(%a, %b : memref<10xf32>, memref<10x10xf32>) {
      acc.yield
    }
    func.return
  }
  // CHECK: acc.serial firstprivate(%{{.*}} : memref<10xf32>) private(%{{.*}}, %{{.*}} : memref<10xf32>, memref<10x10xf32>) {

  func.func @acc_serial_data_operands(%a: memref<10xf32>) {
    acc.serial dataOperands(%a : memref<10xf32>) {
      acc.yield
    }
    func.return
  }
  // CHECK: acc.serial dataOperands(%{{.*}} : memref<10xf32>) {

  func.func @acc_serial_reduction(%a: memref<10xf32>) {
    acc.serial reduction(%a : memref<10xf32>) {
      acc.yield
    }
    func.return
  }
  // CHECK: acc.serial reduction(%{{.*}} : memref<10xf32>) {

  func.func @acc_serial_if_self(%cond: i1) {
    acc.serial self(%cond) if(%cond) {
      acc.yield
    }
    func.return
  }
  // CHECK: acc.serial self(%{{.*}}) if(%{{.*}}) {

  func.func @acc_serial_combined() {
    acc.serial combined(loop) {
      acc.yield
    }
    func.return
  }
  // CHECK: acc.serial combined(loop) {

  func.func @acc_serial_default_attr_present() {
    acc.serial {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>}
    func.return
  }
  // CHECK: acc.serial {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

  func.func @acc_serial_default_attr_none() {
    acc.serial {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue none>}
    func.return
  }
  // CHECK: attributes {defaultAttr = #acc<defaultvalue none>}

  func.func @acc_serial_self_attr() {
    acc.serial {
      acc.yield
    } attributes {selfAttr}
    func.return
  }
  // CHECK: acc.serial {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } attributes {selfAttr}
}
