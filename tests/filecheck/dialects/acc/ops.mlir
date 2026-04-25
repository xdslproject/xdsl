// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func @empty() {
    acc.parallel {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @empty() {
  // CHECK-NEXT:    acc.parallel {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @if_self(%c1 : i1) {
    acc.parallel self(%c1) if(%c1) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @if_self(
  // CHECK:         acc.parallel self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @data_ops(%m : memref<10xf32>) {
    acc.parallel dataOperands(%m : memref<10xf32>) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @data_ops(
  // CHECK:         acc.parallel dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @private_firstprivate_reduction(%m : memref<10xf32>) {
    acc.parallel firstprivate(%m : memref<10xf32>) private(%m : memref<10xf32>) reduction(%m : memref<10xf32>) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @private_firstprivate_reduction(
  // CHECK:         acc.parallel firstprivate(%{{.*}} : memref<10xf32>) private(%{{.*}} : memref<10xf32>) reduction(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @async_bare() {
    acc.parallel async {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @async_bare() {
  // CHECK-NEXT:    acc.parallel async {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @async_keyword_only_with_dt() {
    acc.parallel async([#acc.device_type<nvidia>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @async_keyword_only_with_dt() {
  // CHECK-NEXT:    acc.parallel async([#acc.device_type<nvidia>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @async_one_operand(%a : i64) {
    acc.parallel async(%a : i64) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @async_one_operand(
  // CHECK:         acc.parallel async(%{{.*}} : i64) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @async_operand_with_dt(%a : i64) {
    acc.parallel async(%a : i64 [#acc.device_type<nvidia>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @async_operand_with_dt(
  // CHECK:         acc.parallel async(%{{.*}} : i64 [#acc.device_type<nvidia>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @async_mixed(%a : i64) {
    acc.parallel async([#acc.device_type<nvidia>], %a : i64 [#acc.device_type<default>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @async_mixed(
  // CHECK:         acc.parallel async([#acc.device_type<nvidia>], %{{.*}} : i64 [#acc.device_type<default>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @num_gangs_single(%a : i32) {
    acc.parallel num_gangs({%a : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @num_gangs_single(
  // CHECK:         acc.parallel num_gangs({%{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @num_gangs_multi(%a : i32, %b : i32, %c : index) {
    acc.parallel num_gangs({%a : i32, %b : i32, %c : index}) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @num_gangs_multi(
  // CHECK:         acc.parallel num_gangs({%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : index}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @num_gangs_multi_dt(%a : i32, %b : i32, %c : i32) {
    acc.parallel num_gangs({%a : i32} [#acc.device_type<default>], {%b : i32, %c : i32} [#acc.device_type<nvidia>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @num_gangs_multi_dt(
  // CHECK:         acc.parallel num_gangs({%{{.*}} : i32} [#acc.device_type<default>], {%{{.*}} : i32, %{{.*}} : i32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @num_workers_vector_length(%a : i64, %b : i32) {
    acc.parallel num_workers(%a : i64 [#acc.device_type<nvidia>]) vector_length(%b : i32) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @num_workers_vector_length(
  // CHECK:         acc.parallel num_workers(%{{.*}} : i64 [#acc.device_type<nvidia>]) vector_length(%{{.*}} : i32) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_one_operand(%a : i64, %b : i32) {
    acc.parallel wait(%a, %b : i64, i32) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @wait_one_operand(
  // CHECK:         acc.parallel wait(%{{.*}}, %{{.*}} : i64, i32) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @combined_prefix() {
    acc.parallel combined(loop) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @combined_prefix() {
  // CHECK-NEXT:    acc.parallel combined(loop) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @default_self_attrs_via_attr_dict() {
    acc.parallel {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK-LABEL: func.func @default_self_attrs_via_attr_dict() {
  // CHECK-NEXT:    acc.parallel {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  func.func @generic_roundtrip_retained() {
    "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  // CHECK-LABEL: func.func @generic_roundtrip_retained() {
  // CHECK-NEXT:    acc.parallel {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }
}
