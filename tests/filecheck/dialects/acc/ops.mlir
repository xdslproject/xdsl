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
}
