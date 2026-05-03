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

  func.func @wait_bare() {
    acc.parallel wait {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @wait_bare() {
  // CHECK-NEXT:    acc.parallel wait {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_keyword_only_dt(%a : i64) {
    acc.parallel wait([#acc.device_type<nvidia>], {%a : i64}) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @wait_keyword_only_dt(
  // CHECK:         acc.parallel wait([#acc.device_type<nvidia>], {%{{.*}} : i64}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_group(%a : i64, %b : i32) {
    acc.parallel wait({%a : i64, %b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @wait_group(
  // CHECK:         acc.parallel wait({%{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_devnum(%a : i64, %b : i32) {
    acc.parallel wait({devnum: %a : i64, %b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @wait_devnum(
  // CHECK:         acc.parallel wait({devnum: %{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_devnum_multi(%a : i64, %b : i32) {
    acc.parallel wait({devnum: %a : i64}, {%b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @wait_devnum_multi(
  // CHECK:         acc.parallel wait({devnum: %{{.*}} : i64}, {%{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_mixed(%a : i64, %b : i32) {
    acc.parallel wait([#acc.device_type<nvidia>], {devnum: %a : i64, %b : i32} [#acc.device_type<default>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @wait_mixed(
  // CHECK:         acc.parallel wait([#acc.device_type<nvidia>], {devnum: %{{.*}} : i64, %{{.*}} : i32} [#acc.device_type<default>]) {
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

  func.func @serial_empty() {
    acc.serial {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_empty() {
  // CHECK-NEXT:    acc.serial {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_if_self(%c1 : i1) {
    acc.serial self(%c1) if(%c1) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_if_self(
  // CHECK:         acc.serial self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_data_ops(%m : memref<10xf32>) {
    acc.serial dataOperands(%m : memref<10xf32>) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_data_ops(
  // CHECK:         acc.serial dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_private_firstprivate_reduction(%m : memref<10xf32>) {
    acc.serial firstprivate(%m : memref<10xf32>) private(%m : memref<10xf32>) reduction(%m : memref<10xf32>) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_private_firstprivate_reduction(
  // CHECK:         acc.serial firstprivate(%{{.*}} : memref<10xf32>) private(%{{.*}} : memref<10xf32>) reduction(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_async_bare() {
    acc.serial async {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_async_bare() {
  // CHECK-NEXT:    acc.serial async {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_async_keyword_only_with_dt() {
    acc.serial async([#acc.device_type<nvidia>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_async_keyword_only_with_dt() {
  // CHECK-NEXT:    acc.serial async([#acc.device_type<nvidia>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_async_one_operand(%a : i64) {
    acc.serial async(%a : i64) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_async_one_operand(
  // CHECK:         acc.serial async(%{{.*}} : i64) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_async_operand_with_dt(%a : i64) {
    acc.serial async(%a : i64 [#acc.device_type<nvidia>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_async_operand_with_dt(
  // CHECK:         acc.serial async(%{{.*}} : i64 [#acc.device_type<nvidia>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_async_mixed(%a : i64) {
    acc.serial async([#acc.device_type<nvidia>], %a : i64 [#acc.device_type<default>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_async_mixed(
  // CHECK:         acc.serial async([#acc.device_type<nvidia>], %{{.*}} : i64 [#acc.device_type<default>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_wait_bare() {
    acc.serial wait {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_wait_bare() {
  // CHECK-NEXT:    acc.serial wait {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_wait_group(%a : i64, %b : i32) {
    acc.serial wait({%a : i64, %b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_wait_group(
  // CHECK:         acc.serial wait({%{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_wait_devnum(%a : i64, %b : i32) {
    acc.serial wait({devnum: %a : i64, %b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_wait_devnum(
  // CHECK:         acc.serial wait({devnum: %{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_wait_mixed(%a : i64, %b : i32) {
    acc.serial wait([#acc.device_type<nvidia>], {devnum: %a : i64, %b : i32} [#acc.device_type<default>]) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_wait_mixed(
  // CHECK:         acc.serial wait([#acc.device_type<nvidia>], {devnum: %{{.*}} : i64, %{{.*}} : i32} [#acc.device_type<default>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_combined_prefix() {
    acc.serial combined(loop) {
      acc.yield
    }
    func.return
  }
  // CHECK-LABEL: func.func @serial_combined_prefix() {
  // CHECK-NEXT:    acc.serial combined(loop) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_default_self_attrs_via_attr_dict() {
    acc.serial {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK-LABEL: func.func @serial_default_self_attrs_via_attr_dict() {
  // CHECK-NEXT:    acc.serial {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  func.func @serial_generic_roundtrip_retained() {
    "acc.serial"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : () -> ()
    func.return
  }
  // CHECK-LABEL: func.func @serial_generic_roundtrip_retained() {
  // CHECK-NEXT:    acc.serial {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  // Generic-form input with wait operands but none of the wait property
  // attributes (waitOperandsDeviceType / waitOperandsSegments / hasWaitDevnum)
  // exercises the WaitClause printer's fallback path; the pretty parser would
  // never produce this combination because it always populates all three.
  func.func @serial_wait_printer_fallback(%a : i32, %b : i32) {
    "acc.serial"(%a, %b) <{operandSegmentSizes = array<i32: 0, 2, 0, 0, 0, 0, 0, 0>}> ({
      "acc.yield"() : () -> ()
    }) : (i32, i32) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @serial_wait_printer_fallback(
  // CHECK:         acc.serial wait({%{{.*}} : i32, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  // acc.kernels uses `SingleBlockImplicitTerminator(TerminatorOp)`: the
  // pretty-form parser auto-inserts `acc.terminator` when the body is
  // written as `{ }` and the printer elides it again on output. acc.yield
  // is *not* a valid terminator inside acc.kernels (upstream's acc.yield
  // ParentOneOf list excludes KernelsOp).
  func.func @kernels_empty() {
    acc.kernels {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_empty() {
  // CHECK-NEXT:    acc.kernels {
  // CHECK-NEXT:    }

  func.func @kernels_if_self(%c1 : i1) {
    acc.kernels self(%c1) if(%c1) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_if_self(
  // CHECK:         acc.kernels self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:    }

  func.func @kernels_data_ops(%m : memref<10xf32>) {
    acc.kernels dataOperands(%m : memref<10xf32>) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_data_ops(
  // CHECK:         acc.kernels dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    }

  func.func @kernels_private_firstprivate_reduction(%m : memref<10xf32>) {
    acc.kernels firstprivate(%m : memref<10xf32>) private(%m : memref<10xf32>) reduction(%m : memref<10xf32>) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_private_firstprivate_reduction(
  // CHECK:         acc.kernels firstprivate(%{{.*}} : memref<10xf32>) private(%{{.*}} : memref<10xf32>) reduction(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    }

  func.func @kernels_async_bare() {
    acc.kernels async {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_async_bare() {
  // CHECK-NEXT:    acc.kernels async {
  // CHECK-NEXT:    }

  func.func @kernels_async_keyword_only_with_dt() {
    acc.kernels async([#acc.device_type<nvidia>]) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_async_keyword_only_with_dt() {
  // CHECK-NEXT:    acc.kernels async([#acc.device_type<nvidia>]) {
  // CHECK-NEXT:    }

  func.func @kernels_async_one_operand(%a : i64) {
    acc.kernels async(%a : i64) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_async_one_operand(
  // CHECK:         acc.kernels async(%{{.*}} : i64) {
  // CHECK-NEXT:    }

  func.func @kernels_async_operand_with_dt(%a : i64) {
    acc.kernels async(%a : i64 [#acc.device_type<nvidia>]) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_async_operand_with_dt(
  // CHECK:         acc.kernels async(%{{.*}} : i64 [#acc.device_type<nvidia>]) {
  // CHECK-NEXT:    }

  func.func @kernels_async_mixed(%a : i64) {
    acc.kernels async([#acc.device_type<nvidia>], %a : i64 [#acc.device_type<default>]) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_async_mixed(
  // CHECK:         acc.kernels async([#acc.device_type<nvidia>], %{{.*}} : i64 [#acc.device_type<default>]) {
  // CHECK-NEXT:    }

  func.func @kernels_num_gangs_single(%a : i32) {
    acc.kernels num_gangs({%a : i32}) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_num_gangs_single(
  // CHECK:         acc.kernels num_gangs({%{{.*}} : i32}) {
  // CHECK-NEXT:    }

  func.func @kernels_num_gangs_multi(%a : i32, %b : i32, %c : index) {
    acc.kernels num_gangs({%a : i32, %b : i32, %c : index}) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_num_gangs_multi(
  // CHECK:         acc.kernels num_gangs({%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : index}) {
  // CHECK-NEXT:    }

  func.func @kernels_num_gangs_multi_dt(%a : i32, %b : i32, %c : i32) {
    acc.kernels num_gangs({%a : i32} [#acc.device_type<default>], {%b : i32, %c : i32} [#acc.device_type<nvidia>]) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_num_gangs_multi_dt(
  // CHECK:         acc.kernels num_gangs({%{{.*}} : i32} [#acc.device_type<default>], {%{{.*}} : i32, %{{.*}} : i32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT:    }

  func.func @kernels_num_workers_vector_length(%a : i64, %b : i32) {
    acc.kernels num_workers(%a : i64 [#acc.device_type<nvidia>]) vector_length(%b : i32) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_num_workers_vector_length(
  // CHECK:         acc.kernels num_workers(%{{.*}} : i64 [#acc.device_type<nvidia>]) vector_length(%{{.*}} : i32) {
  // CHECK-NEXT:    }

  func.func @kernels_wait_bare() {
    acc.kernels wait {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_wait_bare() {
  // CHECK-NEXT:    acc.kernels wait {
  // CHECK-NEXT:    }

  func.func @kernels_wait_group(%a : i64, %b : i32) {
    acc.kernels wait({%a : i64, %b : i32}) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_wait_group(
  // CHECK:         acc.kernels wait({%{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:    }

  func.func @kernels_wait_devnum(%a : i64, %b : i32) {
    acc.kernels wait({devnum: %a : i64, %b : i32}) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_wait_devnum(
  // CHECK:         acc.kernels wait({devnum: %{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:    }

  func.func @kernels_wait_mixed(%a : i64, %b : i32) {
    acc.kernels wait([#acc.device_type<nvidia>], {devnum: %a : i64, %b : i32} [#acc.device_type<default>]) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_wait_mixed(
  // CHECK:         acc.kernels wait([#acc.device_type<nvidia>], {devnum: %{{.*}} : i64, %{{.*}} : i32} [#acc.device_type<default>]) {
  // CHECK-NEXT:    }

  func.func @kernels_combined_prefix() {
    acc.kernels combined(loop) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @kernels_combined_prefix() {
  // CHECK-NEXT:    acc.kernels combined(loop) {
  // CHECK-NEXT:    }

  func.func @kernels_default_self_attrs_via_attr_dict() {
    acc.kernels {
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK-LABEL: func.func @kernels_default_self_attrs_via_attr_dict() {
  // CHECK-NEXT:    acc.kernels {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  func.func @kernels_generic_roundtrip_retained() {
    "acc.kernels"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
    }) : () -> ()
    func.return
  }
  // CHECK-LABEL: func.func @kernels_generic_roundtrip_retained() {
  // CHECK-NEXT:    acc.kernels {
  // CHECK-NEXT:    }

  // acc.data — structured data construct. Body uses acc.terminator (same
  // SingleBlockImplicitTerminator(TerminatorOp) shape as kernels). The
  // verifier additionally requires either at least one operand or
  // `defaultAttr`, so a body-only case must carry one of them.
  func.func @data_default_attr() {
    acc.data {
    } attributes {defaultAttr = #acc<defaultvalue none>}
    func.return
  }
  // CHECK-LABEL: func.func @data_default_attr() {
  // CHECK-NEXT:    acc.data {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue none>}

  func.func @data_data_operand(%a : memref<10xf32>) {
    %p = acc.present varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.data dataOperands(%p : memref<10xf32>) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @data_data_operand(
  // CHECK:         acc.data dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    }

  func.func @data_if_async_wait(%c : i1, %a : i64, %w : i64) {
    acc.data if(%c) async(%a : i64) wait({%w : i64}) {
    } attributes {defaultAttr = #acc<defaultvalue none>}
    func.return
  }
  // CHECK-LABEL: func.func @data_if_async_wait(
  // CHECK:         acc.data if(%{{.*}}) async(%{{.*}} : i64) wait({%{{.*}} : i64}) {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue none>}

  func.func @data_async_bare() {
    acc.data async {
    } attributes {defaultAttr = #acc<defaultvalue none>}
    func.return
  }
  // CHECK-LABEL: func.func @data_async_bare() {
  // CHECK-NEXT:    acc.data async {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue none>}

  // Generic-form roundtrip retains insurance that the operandSegmentSizes
  // (4 segments: ifCond / async / wait / data) round-trip.
  func.func @data_generic_roundtrip_retained() {
    "acc.data"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0>}> ({
    }) {defaultAttr = #acc<defaultvalue none>} : () -> ()
    func.return
  }
  // CHECK-LABEL: func.func @data_generic_roundtrip_retained() {
  // CHECK-NEXT:    acc.data {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue none>}

  // acc.host_data — like acc.data but the operands must be defined by
  // acc.use_device, and there is an `ifPresent` UnitAttr instead of a
  // default clause.
  func.func @host_data_minimal(%a : memref<10xf32>) {
    %u = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.host_data dataOperands(%u : memref<10xf32>) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @host_data_minimal(
  // CHECK:         acc.host_data dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    }

  func.func @host_data_if_present(%a : memref<10xf32>) {
    %u = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.host_data dataOperands(%u : memref<10xf32>) {
    } attributes {ifPresent}
    func.return
  }
  // CHECK-LABEL: func.func @host_data_if_present(
  // CHECK:         acc.host_data dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    } attributes {ifPresent}

  func.func @host_data_if_cond(%c : i1, %a : memref<10xf32>) {
    %u = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.host_data if(%c) dataOperands(%u : memref<10xf32>) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @host_data_if_cond(
  // CHECK:         acc.host_data if(%{{.*}}) dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    }

  // Generic-form roundtrip insurance: 2 segments (ifCond / data) and the
  // ifPresent UnitAttr ride in the trailing attr-dict.
  func.func @host_data_generic_roundtrip_retained(%a : memref<10xf32>) {
    %u = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    "acc.host_data"(%u) <{operandSegmentSizes = array<i32: 0, 1>, ifPresent}> ({
    }) : (memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @host_data_generic_roundtrip_retained(
  // CHECK:         acc.host_data dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    } attributes {ifPresent}

  func.func @bounds_full(%c0 : index, %c9 : index, %c1 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index) stride(%c1 : index)
    func.return
  }
  // CHECK-LABEL: func.func @bounds_full(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index)

  func.func @bounds_extent_only(%c9 : index) {
    %b = acc.bounds extent(%c9 : index)
    func.return
  }
  // CHECK-LABEL: func.func @bounds_extent_only(
  // CHECK:         %{{.*}} = acc.bounds extent(%{{.*}} : index)

  func.func @bounds_upperbound_only(%c9 : index) {
    %b = acc.bounds upperbound(%c9 : index)
    func.return
  }
  // CHECK-LABEL: func.func @bounds_upperbound_only(
  // CHECK:         %{{.*}} = acc.bounds upperbound(%{{.*}} : index)

  func.func @bounds_with_stride_in_bytes(%c1 : index, %c20 : index, %c4 : index) {
    %b = acc.bounds lowerbound(%c1 : index) upperbound(%c20 : index) extent(%c20 : index) stride(%c4 : index) startIdx(%c1 : index) {strideInBytes = true}
    func.return
  }
  // CHECK-LABEL: func.func @bounds_with_stride_in_bytes(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index) {strideInBytes = true}

  // Generic form is retained as insurance that the operandSegmentSizes
  // round-trip (with all clauses absent except `extent`) still parses.
  func.func @bounds_generic_roundtrip_retained(%c9 : index) {
    %b = "acc.bounds"(%c9) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> : (index) -> !acc.data_bounds_ty
    func.return
  }
  // CHECK-LABEL: func.func @bounds_generic_roundtrip_retained(
  // CHECK:         %{{.*}} = acc.bounds extent(%{{.*}} : index)

  func.func @bounds_accessors(%b : !acc.data_bounds_ty) {
    %lb = acc.get_lowerbound %b : (!acc.data_bounds_ty) -> index
    %ub = acc.get_upperbound %b : (!acc.data_bounds_ty) -> index
    %stride = acc.get_stride %b : (!acc.data_bounds_ty) -> index
    %extent = acc.get_extent %b : (!acc.data_bounds_ty) -> index
    func.return
  }
  // CHECK-LABEL: func.func @bounds_accessors(
  // CHECK:         %{{.*}} = acc.get_lowerbound %{{.*}} : (!acc.data_bounds_ty) -> index
  // CHECK-NEXT:    %{{.*}} = acc.get_upperbound %{{.*}} : (!acc.data_bounds_ty) -> index
  // CHECK-NEXT:    %{{.*}} = acc.get_stride %{{.*}} : (!acc.data_bounds_ty) -> index
  // CHECK-NEXT:    %{{.*}} = acc.get_extent %{{.*}} : (!acc.data_bounds_ty) -> index

  // Entry data-clause ops sharing the abstract `_DataEntryOperation` mixin:
  // copyin / create / present (more leaves follow). All three share the same operand
  // and property surface; cover it thoroughly via `acc.copyin` and
  // spot-check the per-op `dataClause` default elision via `acc.create` /
  // `acc.present`.
  func.func @copyin_minimal(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_minimal(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @copyin_var_keyword(%a : memref<10xf32>) {
    // The `var` keyword is also accepted on parse (the printer always emits
    // `varPtr`). Exercises the alternative branch in `Var.parse`.
    %r = acc.copyin var(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_var_keyword(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @copyin_with_var_type(%a : memref<10xf32>) {
    // varType differs from the var's element type, so `Var.print` emits the
    // optional `varType(...)` slot.
    %r = acc.copyin varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_with_var_type(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>

  func.func @copyin_with_var_ptr_ptr(%a : memref<10xf32>, %p : memref<memref<10xf32>>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) varPtrPtr(%p : memref<memref<10xf32>>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_with_var_ptr_ptr(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) varPtrPtr(%{{.*}} : memref<memref<10xf32>>) -> memref<10xf32>

  func.func @copyin_with_bounds(%a : memref<10xf32>, %c0 : index, %c9 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
    %r = acc.copyin varPtr(%a : memref<10xf32>) bounds(%b) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_with_bounds(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index)
  // CHECK-NEXT:    %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) bounds(%{{.*}}) -> memref<10xf32>

  func.func @copyin_async_bare(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) async -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_async_bare(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) async -> memref<10xf32>

  func.func @copyin_async_kw_dt(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) async([#acc.device_type<nvidia>]) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_async_kw_dt(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) async([#acc.device_type<nvidia>]) -> memref<10xf32>

  func.func @copyin_async_operand(%a : memref<10xf32>, %async : i32) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) async(%async : i32) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_async_operand(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) async(%{{.*}} : i32) -> memref<10xf32>

  func.func @copyin_async_operand_dt(%a : memref<10xf32>, %async : i32) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) async(%async : i32 [#acc.device_type<nvidia>]) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_async_operand_dt(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) async(%{{.*}} : i32 [#acc.device_type<nvidia>]) -> memref<10xf32>

  func.func @copyin_full_attr_dict(%a : memref<10xf32>) {
    // Every defaulted prop overridden in attr-dict, to verify each round-trips
    // (rather than collapsing to the per-op default and silently disappearing).
    %r = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyin_readonly>, implicit = true, modifiers = #acc<data_clause_modifier readonly>, name = "myvar", structured = false}
    func.return
  }
  // CHECK-LABEL: func.func @copyin_full_attr_dict(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyin_readonly>, implicit = true, modifiers = #acc<data_clause_modifier readonly>, name = "myvar", structured = false}

  // Generic-form input retained per dialect convention — insurance that
  // operandSegmentSizes round-trips through the generic surface even after
  // the pretty form lands.
  func.func @copyin_generic_roundtrip(%a : memref<10xf32>) {
    %r = "acc.copyin"(%a) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_generic_roundtrip(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @create_minimal(%a : memref<10xf32>) {
    // No dataClause in attr-dict: matches the per-op default (acc_create),
    // so attr-dict elision suppresses it on print.
    %r = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @create_minimal(
  // CHECK:         %{{.*}} = acc.create varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @create_with_clause_override(%a : memref<10xf32>) {
    // `acc.create` decomposed from `copyout` keeps the original clause name
    // in its `dataClause` attr — non-default value, so attr-dict prints it.
    %r = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout>}
    func.return
  }
  // CHECK-LABEL: func.func @create_with_clause_override(
  // CHECK:         %{{.*}} = acc.create varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout>}

  func.func @present_minimal(%a : memref<10xf32>) {
    %r = acc.present varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @present_minimal(
  // CHECK:         %{{.*}} = acc.present varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  // Non-memref var: `_default_var_type` falls back to the var's own type
  // (the upstream `printVarPtrType` heuristic for non-pointer-like types).
  // Both parser-side defaulting and printer-side elision exercised here.
  // mlir-opt rejects this case (its operand constraint requires a pointer-
  // like or mappable type), so it lives only in the xDSL-only roundtrip.
  func.func @copyin_non_memref_var(%a : i32) {
    %r = acc.copyin varPtr(%a : i32) -> i32
    func.return
  }
  // CHECK-LABEL: func.func @copyin_non_memref_var(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : i32) -> i32

  // Result type differs from the var's type: the assembly format's
  // `type($acc_var)` slot is honored independently of `var.type`.
  func.func @copyin_explicit_acc_var_type(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) -> memref<20xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_explicit_acc_var_type(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) -> memref<20xf32>

  // Per-op `dataClause` default wiring. Generic-form input carries the
  // default value explicitly; the pretty-form output must elide it. If a
  // leaf class were wired to the wrong default, the round-trip would
  // either keep the attr (when input differs from the leaf's actual
  // default) or drop the wrong one — either way visible in the CHECK.
  func.func @copyin_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.copyin"(%a) <{dataClause = #acc<data_clause acc_copyin>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @create_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.create"(%a) <{dataClause = #acc<data_clause acc_create>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @create_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.create varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @present_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.present"(%a) <{dataClause = #acc<data_clause acc_present>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @present_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.present varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  // Remaining seven entry data-clause leaves. Each gets one minimal pretty-
  // form roundtrip and one generic-form input that carries the leaf's
  // `dataClause` default explicitly — pretty-form output must elide it,
  // proving the per-op default is wired correctly.
  func.func @nocreate_minimal(%a : memref<10xf32>) {
    %r = acc.nocreate varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @nocreate_minimal(
  // CHECK:         %{{.*}} = acc.nocreate varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @nocreate_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.nocreate"(%a) <{dataClause = #acc<data_clause acc_no_create>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @nocreate_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.nocreate varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @attach_minimal(%a : memref<10xf32>) {
    %r = acc.attach varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @attach_minimal(
  // CHECK:         %{{.*}} = acc.attach varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @attach_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.attach"(%a) <{dataClause = #acc<data_clause acc_attach>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @attach_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.attach varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @deviceptr_minimal(%a : memref<10xf32>) {
    %r = acc.deviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @deviceptr_minimal(
  // CHECK:         %{{.*}} = acc.deviceptr varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @deviceptr_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.deviceptr"(%a) <{dataClause = #acc<data_clause acc_deviceptr>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @deviceptr_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.deviceptr varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @use_device_minimal(%a : memref<10xf32>) {
    %r = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @use_device_minimal(
  // CHECK:         %{{.*}} = acc.use_device varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @use_device_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.use_device"(%a) <{dataClause = #acc<data_clause acc_use_device>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @use_device_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.use_device varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @cache_minimal(%a : memref<10xf32>) {
    %r = acc.cache varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @cache_minimal(
  // CHECK:         %{{.*}} = acc.cache varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @cache_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.cache"(%a) <{dataClause = #acc<data_clause acc_cache>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @cache_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.cache varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @declare_device_resident_minimal(%a : memref<10xf32>) {
    %r = acc.declare_device_resident varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @declare_device_resident_minimal(
  // CHECK:         %{{.*}} = acc.declare_device_resident varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @declare_device_resident_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.declare_device_resident"(%a) <{dataClause = #acc<data_clause acc_declare_device_resident>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @declare_device_resident_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.declare_device_resident varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @declare_link_minimal(%a : memref<10xf32>) {
    %r = acc.declare_link varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @declare_link_minimal(
  // CHECK:         %{{.*}} = acc.declare_link varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @declare_link_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.declare_link"(%a) <{dataClause = #acc<data_clause acc_declare_link>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @declare_link_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.declare_link varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @copyout_minimal(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.copyout accPtr(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @copyout_minimal(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>)

  // The `AccVar` directive accepts either `accPtr` or `accVar` on parse and
  // always emits `accPtr` on print (mirroring `Var`'s `varPtr` choice). Input
  // uses the `accVar` spelling; the output must normalize to `accPtr`,
  // proving the parser fallback is reachable.
  func.func @copyout_acc_var_keyword(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.copyout accVar(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @copyout_acc_var_keyword(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @copyout_with_var_type(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.copyout accPtr(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>) varType(tensor<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @copyout_with_var_type(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>) varType(tensor<10xf32>)

  func.func @copyout_with_bounds(%d : memref<10xf32>, %h : memref<10xf32>, %c0 : index, %c9 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
    acc.copyout accPtr(%d : memref<10xf32>) bounds(%b) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @copyout_with_bounds(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index)
  // CHECK-NEXT:    acc.copyout accPtr(%{{.*}} : memref<10xf32>) bounds(%{{.*}}) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @copyout_async_bare(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.copyout accPtr(%d : memref<10xf32>) async to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @copyout_async_bare(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) async to varPtr(%{{.*}} : memref<10xf32>)

  func.func @copyout_async_operand(%d : memref<10xf32>, %h : memref<10xf32>, %async : i32) {
    acc.copyout accPtr(%d : memref<10xf32>) async(%async : i32) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @copyout_async_operand(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) async(%{{.*}} : i32) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @copyout_async_operand_dt(%d : memref<10xf32>, %h : memref<10xf32>, %async : i32) {
    acc.copyout accPtr(%d : memref<10xf32>) async(%async : i32 [#acc.device_type<nvidia>]) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @copyout_async_operand_dt(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) async(%{{.*}} : i32 [#acc.device_type<nvidia>]) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @copyout_full_attr_dict(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.copyout accPtr(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>) {dataClause = #acc<data_clause acc_copyout_zero>, implicit = true, modifiers = #acc<data_clause_modifier zero>, name = "myvar", structured = false}
    func.return
  }
  // CHECK-LABEL: func.func @copyout_full_attr_dict(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>) {dataClause = #acc<data_clause acc_copyout_zero>, implicit = true, modifiers = #acc<data_clause_modifier zero>, name = "myvar", structured = false}

  // Generic-form roundtrip insurance for `acc.copyout` — proves the four-
  // group `operandSegmentSizes` shape (`accVar`, `var`, `bounds`,
  // `asyncOperands`) survives the generic surface even after the pretty
  // form lands.
  func.func @copyout_generic_roundtrip(%d : memref<10xf32>, %h : memref<10xf32>) {
    "acc.copyout"(%d, %h) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, varType = f32}> : (memref<10xf32>, memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @copyout_generic_roundtrip(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>)

  // Per-op `dataClause` default elision for each exit leaf — generic-form
  // input carries the leaf's default explicitly; pretty-form output must
  // elide it. Catches a wrong-default wiring per leaf.
  func.func @copyout_dataclause_default_elided(%d : memref<10xf32>, %h : memref<10xf32>) {
    "acc.copyout"(%d, %h) <{dataClause = #acc<data_clause acc_copyout>, operandSegmentSizes = array<i32: 1, 1, 0, 0>, varType = f32}> : (memref<10xf32>, memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @copyout_dataclause_default_elided(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>)
  // CHECK-NOT:     dataClause

  func.func @update_host_minimal(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.update_host accPtr(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @update_host_minimal(
  // CHECK:         acc.update_host accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @update_host_dataclause_default_elided(%d : memref<10xf32>, %h : memref<10xf32>) {
    "acc.update_host"(%d, %h) <{dataClause = #acc<data_clause acc_update_host>, operandSegmentSizes = array<i32: 1, 1, 0, 0>, varType = f32}> : (memref<10xf32>, memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @update_host_dataclause_default_elided(
  // CHECK:         acc.update_host accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>)
  // CHECK-NOT:     dataClause

  // The two entry-shape leaves — they live on `_DataEntryOperation`
  // (same shape as `acc.copyin`), they just differ in the per-op
  // `dataClause` default.
  func.func @getdeviceptr_minimal(%a : memref<10xf32>) {
    %r = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @getdeviceptr_minimal(
  // CHECK:         %{{.*}} = acc.getdeviceptr varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @getdeviceptr_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.getdeviceptr"(%a) <{dataClause = #acc<data_clause acc_getdeviceptr>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @getdeviceptr_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.getdeviceptr varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @update_device_minimal(%a : memref<10xf32>) {
    %r = acc.update_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @update_device_minimal(
  // CHECK:         %{{.*}} = acc.update_device varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @update_device_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.update_device"(%a) <{dataClause = #acc<data_clause acc_update_device>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @update_device_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.update_device varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  // No-host-pointer exit ops (`acc.delete`, `acc.detach`). These live on
  // `_DataExitOperationNoVarPtr` — three operand groups (`accVar`,
  // `bounds`, `asyncOperands`), no host `var`, no `varType`. They share
  // the `AccVar` custom directive and the bounds/async assembly format
  // tail with the with-host-pointer exit ops.
  func.func @delete_minimal(%d : memref<10xf32>) {
    acc.delete accPtr(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @delete_minimal(
  // CHECK:         acc.delete accPtr(%{{.*}} : memref<10xf32>)

  func.func @delete_with_bounds_async(%d : memref<10xf32>, %async : i32, %c0 : index, %c9 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
    acc.delete accPtr(%d : memref<10xf32>) bounds(%b) async(%async : i32)
    func.return
  }
  // CHECK-LABEL: func.func @delete_with_bounds_async(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index)
  // CHECK-NEXT:    acc.delete accPtr(%{{.*}} : memref<10xf32>) bounds(%{{.*}}) async(%{{.*}} : i32)

  func.func @delete_async_kw_dt(%d : memref<10xf32>) {
    acc.delete accPtr(%d : memref<10xf32>) async([#acc.device_type<nvidia>])
    func.return
  }
  // CHECK-LABEL: func.func @delete_async_kw_dt(
  // CHECK:         acc.delete accPtr(%{{.*}} : memref<10xf32>) async([#acc.device_type<nvidia>])

  func.func @delete_full_attr_dict(%d : memref<10xf32>) {
    acc.delete accPtr(%d : memref<10xf32>) {dataClause = #acc<data_clause acc_create>, implicit = true, modifiers = #acc<data_clause_modifier alwaysout>, name = "myvar", structured = false}
    func.return
  }
  // CHECK-LABEL: func.func @delete_full_attr_dict(
  // CHECK:         acc.delete accPtr(%{{.*}} : memref<10xf32>) {dataClause = #acc<data_clause acc_create>, implicit = true, modifiers = #acc<data_clause_modifier alwaysout>, name = "myvar", structured = false}

  // Generic-form roundtrip insurance for the no-varPtr exit shape. Three
  // operand groups (`accVar`, `bounds`, `asyncOperands`) — distinct from
  // the four-group WithVarPtr shape, so the `operandSegmentSizes` shape is
  // independently load-bearing here.
  func.func @delete_generic_roundtrip(%d : memref<10xf32>) {
    "acc.delete"(%d) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @delete_generic_roundtrip(
  // CHECK:         acc.delete accPtr(%{{.*}} : memref<10xf32>)

  func.func @delete_dataclause_default_elided(%d : memref<10xf32>) {
    "acc.delete"(%d) <{dataClause = #acc<data_clause acc_delete>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @delete_dataclause_default_elided(
  // CHECK:         acc.delete accPtr(%{{.*}} : memref<10xf32>)
  // CHECK-NOT:     dataClause

  func.func @detach_minimal(%d : memref<10xf32>) {
    acc.detach accPtr(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @detach_minimal(
  // CHECK:         acc.detach accPtr(%{{.*}} : memref<10xf32>)

  func.func @detach_dataclause_default_elided(%d : memref<10xf32>) {
    "acc.detach"(%d) <{dataClause = #acc<data_clause acc_detach>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @detach_dataclause_default_elided(
  // CHECK:         acc.detach accPtr(%{{.*}} : memref<10xf32>)
  // CHECK-NOT:     dataClause

  // The inherited `recipe` property is consumed by the `DataEntryOilist`
  // custom directive as an inline `recipe(@sym)` clause — matching
  // upstream's `oilist(... | `recipe` `(` ... `)`)`. Both the inline
  // pretty-form and the attr-dict spelling parse to the same op; the
  // pretty form is canonical on print. Exercised on `acc.copyin` since
  // `_DataEntryOperation`'s assembly format is uniform across every entry
  // leaf.
  func.func @copyin_with_recipe(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) recipe(@some_recipe) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_with_recipe(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) recipe(@some_recipe) -> memref<10xf32>

  // Attr-dict spelling for `recipe` — accepted on parse, normalized to the
  // inline `recipe(@sym)` form on print. Exercises the `ParsePropInAttrDict`
  // fallback for the same property.
  func.func @copyin_recipe_attr_dict_input(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {recipe = @some_recipe}
    func.return
  }
  // CHECK-LABEL: func.func @copyin_recipe_attr_dict_input(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) recipe(@some_recipe) -> memref<10xf32>

  // Order-free input: the `DataEntryOilist` directive accepts the four
  // optional clauses (varPtrPtr / bounds / async / recipe) in any order on
  // parse (mirroring upstream's `oilist(...)` semantics), and emits them
  // in the canonical td-definition order on print. This is what makes the
  // `mlir-opt` round-trip tolerant to upstream's own clause-order choices.
  func.func @copyin_clauses_reordered(%a : memref<10xf32>, %p : memref<memref<10xf32>>, %c0 : index, %c9 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
    %r = acc.copyin varPtr(%a : memref<10xf32>) recipe(@some_recipe) async bounds(%b) varPtrPtr(%p : memref<memref<10xf32>>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @copyin_clauses_reordered(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) varPtrPtr(%{{.*}} : memref<memref<10xf32>>) bounds(%{{.*}}) async recipe(@some_recipe) -> memref<10xf32>

  // Privatization data-clause ops (`acc.private`, `acc.firstprivate`,
  // `acc.firstprivate_map`, `acc.reduction`). These share the entry
  // `_DataEntryOperation` mixin — same operand and assembly-format
  // surface as `acc.copyin` etc. — and differ only in the per-op
  // `dataClause` default. The natural extra coverage here is the
  // `recipe(@sym)` reference case: `acc.private` / `acc.firstprivate` /
  // `acc.reduction` all carry a recipe pointing at one of the
  // privatization / reduction recipe symbol ops introduced in PR 13.
  func.func @private_minimal(%a : memref<10xf32>) {
    %r = acc.private varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @private_minimal(
  // CHECK:         %{{.*}} = acc.private varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @private_with_recipe(%a : memref<10xf32>) {
    %r = acc.private varPtr(%a : memref<10xf32>) recipe(@priv_min) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @private_with_recipe(
  // CHECK:         %{{.*}} = acc.private varPtr(%{{.*}} : memref<10xf32>) recipe(@priv_min) -> memref<10xf32>

  func.func @private_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.private"(%a) <{dataClause = #acc<data_clause acc_private>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @private_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.private varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @firstprivate_minimal(%a : memref<10xf32>) {
    %r = acc.firstprivate varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @firstprivate_minimal(
  // CHECK:         %{{.*}} = acc.firstprivate varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @firstprivate_with_recipe(%a : memref<10xf32>) {
    %r = acc.firstprivate varPtr(%a : memref<10xf32>) recipe(@fp_min) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @firstprivate_with_recipe(
  // CHECK:         %{{.*}} = acc.firstprivate varPtr(%{{.*}} : memref<10xf32>) recipe(@fp_min) -> memref<10xf32>

  func.func @firstprivate_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.firstprivate"(%a) <{dataClause = #acc<data_clause acc_firstprivate>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @firstprivate_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.firstprivate varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @firstprivate_map_minimal(%a : memref<10xf32>) {
    %r = acc.firstprivate_map varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @firstprivate_map_minimal(
  // CHECK:         %{{.*}} = acc.firstprivate_map varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  // `acc.firstprivate_map` decomposes the same user-level
  // `acc_firstprivate` clause as `acc.firstprivate` — its `dataClause`
  // default is `acc_firstprivate` (not the op-name-derived value a typo
  // would land on).
  func.func @firstprivate_map_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.firstprivate_map"(%a) <{dataClause = #acc<data_clause acc_firstprivate>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @firstprivate_map_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.firstprivate_map varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @reduction_minimal(%a : memref<10xf32>) {
    %r = acc.reduction varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @reduction_minimal(
  // CHECK:         %{{.*}} = acc.reduction varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  // `acc.reduction` references the operator via `recipe(@sym)` — the
  // operator itself lives on `acc.reduction.recipe`, not on the data-entry
  // op.
  func.func @reduction_with_recipe(%a : memref<10xf32>) {
    %r = acc.reduction varPtr(%a : memref<10xf32>) recipe(@red_add_i64) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @reduction_with_recipe(
  // CHECK:         %{{.*}} = acc.reduction varPtr(%{{.*}} : memref<10xf32>) recipe(@red_add_i64) -> memref<10xf32>

  func.func @reduction_dataclause_default_elided(%a : memref<10xf32>) {
    %r = "acc.reduction"(%a) <{dataClause = #acc<data_clause acc_reduction>, operandSegmentSizes = array<i32: 1, 0, 0, 0>, varType = f32}> : (memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK-LABEL: func.func @reduction_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.reduction varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  // Privatization recipes — `acc.private.recipe` / `acc.firstprivate.recipe`
  // are top-level Symbol ops (`IsolatedFromAbove`). Each carries a
  // `sym_name` + `type` plus named regions; the `acc.yield`-as-init-result
  // body terminator works because both ops were appended to YieldOp's
  // `HasParent` list.
  acc.private.recipe @priv_min : memref<10xf32> init {
  ^bb0(%arg0: memref<10xf32>):
    acc.yield %arg0 : memref<10xf32>
  }
  // CHECK-LABEL: acc.private.recipe @priv_min : memref<10xf32> init {
  // CHECK-NEXT:    ^{{.*}}(%{{.*}}: memref<10xf32>):
  // CHECK-NEXT:      acc.yield %{{.*}} : memref<10xf32>
  // CHECK-NEXT:    }

  acc.private.recipe @priv_with_destroy : memref<10xf32> init {
  ^bb0(%arg0: memref<10xf32>):
    acc.yield %arg0 : memref<10xf32>
  } destroy {
  ^bb0(%arg0: memref<10xf32>):
    acc.yield
  }
  // CHECK-LABEL: acc.private.recipe @priv_with_destroy : memref<10xf32> init {
  // CHECK:         } destroy {
  // CHECK:           acc.yield
  // CHECK-NEXT:    }

  // Generic-form roundtrip insurance for `acc.private.recipe` — the destroy
  // region is always part of `op.regions` (always printed in generic form),
  // even when it's empty, in which case the optional `destroy` keyword is
  // elided in the pretty form.
  "acc.private.recipe"() <{sym_name = "priv_generic", type = memref<10xf32>}> ({
  ^bb0(%arg0: memref<10xf32>):
    "acc.yield"(%arg0) : (memref<10xf32>) -> ()
  }, {
  }) : () -> ()
  // CHECK-LABEL: acc.private.recipe @priv_generic : memref<10xf32> init {

  acc.firstprivate.recipe @fp_min : memref<10xf32> init {
  ^bb0(%arg0: memref<10xf32>):
    acc.yield %arg0 : memref<10xf32>
  } copy {
  ^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
    acc.yield
  }
  // CHECK-LABEL: acc.firstprivate.recipe @fp_min : memref<10xf32> init {
  // CHECK:         } copy {
  // CHECK-NEXT:    ^{{.*}}(%{{.*}}: memref<10xf32>, %{{.*}}: memref<10xf32>):
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  acc.firstprivate.recipe @fp_with_destroy : memref<10xf32> init {
  ^bb0(%arg0: memref<10xf32>):
    acc.yield %arg0 : memref<10xf32>
  } copy {
  ^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
    acc.yield
  } destroy {
  ^bb0(%arg0: memref<10xf32>):
    acc.yield
  }
  // CHECK-LABEL: acc.firstprivate.recipe @fp_with_destroy : memref<10xf32> init {
  // CHECK:         } copy {
  // CHECK:         } destroy {
  // CHECK-NEXT:    ^{{.*}}(%{{.*}}: memref<10xf32>):
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  // Generic-form roundtrip for the firstprivate three-region shape. An
  // empty trailing region elides `destroy` in pretty form.
  "acc.firstprivate.recipe"() <{sym_name = "fp_generic", type = memref<10xf32>}> ({
  ^bb0(%arg0: memref<10xf32>):
    "acc.yield"(%arg0) : (memref<10xf32>) -> ()
  }, {
  ^bb1(%arg1: memref<10xf32>, %arg2: memref<10xf32>):
    "acc.yield"() : () -> ()
  }, {
  }) : () -> ()
  // CHECK-LABEL: acc.firstprivate.recipe @fp_generic : memref<10xf32> init {

  // Reduction recipe — same Symbol / IsolatedFromAbove shape as the
  // private / firstprivate recipes plus a `reductionOperator` property.
  // The pretty form spells the operator inline as `<add>` (just the enum
  // value), matching upstream's `assemblyFormat = "`<` $value `>`"` on the
  // `ReductionOpKindAttr`. The full `#acc.reduction_operator<add>`
  // opaque form only appears in the generic / attr-dict spellings.
  acc.reduction.recipe @red_add_i64 : i64 reduction_operator <add> init {
  ^bb0(%arg0: i64):
    %c0 = arith.constant 0 : i64
    acc.yield %c0 : i64
  } combiner {
  ^bb0(%arg0: i64, %arg1: i64):
    %r = arith.addi %arg0, %arg1 : i64
    acc.yield %r : i64
  }
  // CHECK-LABEL: acc.reduction.recipe @red_add_i64 : i64 reduction_operator <add> init {
  // CHECK:         } combiner {
  // CHECK-NEXT:    ^{{.*}}(%{{.*}}: i64, %{{.*}}: i64):
  // CHECK:           acc.yield %{{.*}} : i64
  // CHECK-NEXT:    }

  acc.reduction.recipe @red_max_f32 : f32 reduction_operator <max> init {
  ^bb0(%arg0: f32):
    acc.yield %arg0 : f32
  } combiner {
  ^bb0(%arg0: f32, %arg1: f32):
    acc.yield %arg0 : f32
  } destroy {
  ^bb0(%arg0: f32):
    acc.yield
  }
  // CHECK-LABEL: acc.reduction.recipe @red_max_f32 : f32 reduction_operator <max> init {
  // CHECK:         } combiner {
  // CHECK:         } destroy {

  // Generic-form roundtrip insurance for the reduction recipe — the
  // `reductionOperator` rides as `#acc.reduction_operator<add>` in the
  // properties dict (the long opaque form), and the optional `destroy`
  // region is always present in `op.regions` even when empty.
  "acc.reduction.recipe"() <{sym_name = "red_generic", type = i64, reductionOperator = #acc.reduction_operator<add>}> ({
  ^bb0(%arg0: i64):
    "acc.yield"(%arg0) : (i64) -> ()
  }, {
  ^bb1(%arg1: i64, %arg2: i64):
    "acc.yield"(%arg1) : (i64) -> ()
  }, {
  }) : () -> ()
  // CHECK-LABEL: acc.reduction.recipe @red_generic : i64 reduction_operator <add> init {

  // acc.enter_data — standalone unstructured data-entry op. The optional
  // clauses appear in upstream-td-definition order on print:
  //   if, async, wait_devnum, wait, dataOperands.
  // The custom format encodes both the operand-bearing form `async(%v : ty)`
  // and the bare-keyword form `async` (UnitAttr) via the
  // `OperandWithKeywordOnly` directive; same for `wait` via
  // `OperandsWithKeywordOnly`.
  func.func @enter_data_minimal(%a : memref<10xf32>) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @enter_data_minimal(
  // CHECK:         acc.enter_data dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_async_bare(%a : memref<10xf32>) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data async dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @enter_data_async_bare(
  // CHECK:         acc.enter_data async dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_async_operand(%a : memref<10xf32>, %v : i64) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data async(%v : i64) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @enter_data_async_operand(
  // CHECK:         acc.enter_data async(%{{.*}} : i64) dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_wait_bare(%a : memref<10xf32>) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data wait dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @enter_data_wait_bare(
  // CHECK:         acc.enter_data wait dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_if_wait_devnum(%a : memref<10xf32>, %c : i1, %dn : i64, %w : i32) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data if(%c) wait_devnum(%dn : i64) wait(%w : i32) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @enter_data_if_wait_devnum(
  // CHECK:         acc.enter_data if(%{{.*}}) wait_devnum(%{{.*}} : i64) wait(%{{.*}} : i32) dataOperands(%{{.*}} : memref<10xf32>)

  // Generic-form roundtrip insurance for `acc.enter_data` — the
  // `operandSegmentSizes` array gates the five operand groups.
  func.func @enter_data_generic_roundtrip(%a : memref<10xf32>) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    "acc.enter_data"(%d) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 1>}> : (memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @enter_data_generic_roundtrip(
  // CHECK:         acc.enter_data dataOperands(%{{.*}} : memref<10xf32>)

  // acc.exit_data — twin of enter_data plus a `finalize` UnitAttr. Same
  // five-segment operand shape (if, async, wait_devnum, wait, dataOperands)
  // and the same `OperandWithKeywordOnly` / `OperandsWithKeywordOnly`
  // directives. `finalize` rides through `attr-dict-with-keyword`.
  func.func @exit_data_minimal(%a : memref<10xf32>) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @exit_data_minimal(
  // CHECK:         acc.exit_data dataOperands(%{{.*}} : memref<10xf32>)

  func.func @exit_data_async_finalize(%a : memref<10xf32>) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data async dataOperands(%d : memref<10xf32>) attributes {finalize}
    func.return
  }
  // CHECK-LABEL: func.func @exit_data_async_finalize(
  // CHECK:         acc.exit_data async dataOperands(%{{.*}} : memref<10xf32>) attributes {finalize}

  func.func @exit_data_async_operand(%a : memref<10xf32>, %v : i64) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data async(%v : i64) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @exit_data_async_operand(
  // CHECK:         acc.exit_data async(%{{.*}} : i64) dataOperands(%{{.*}} : memref<10xf32>)

  func.func @exit_data_wait_bare(%a : memref<10xf32>) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data wait dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @exit_data_wait_bare(
  // CHECK:         acc.exit_data wait dataOperands(%{{.*}} : memref<10xf32>)

  func.func @exit_data_if_wait_devnum(%a : memref<10xf32>, %c : i1, %dn : i64, %w : i32) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data if(%c) wait_devnum(%dn : i64) wait(%w : i32) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK-LABEL: func.func @exit_data_if_wait_devnum(
  // CHECK:         acc.exit_data if(%{{.*}}) wait_devnum(%{{.*}} : i64) wait(%{{.*}} : i32) dataOperands(%{{.*}} : memref<10xf32>)

  // Generic-form roundtrip insurance for `acc.exit_data` — same five-group
  // operandSegmentSizes shape as enter_data plus the `finalize` UnitAttr.
  func.func @exit_data_generic_roundtrip(%a : memref<10xf32>) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    "acc.exit_data"(%d) <{finalize, operandSegmentSizes = array<i32: 0, 0, 0, 0, 1>}> : (memref<10xf32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @exit_data_generic_roundtrip(
  // CHECK:         acc.exit_data dataOperands(%{{.*}} : memref<10xf32>) attributes {finalize}

  // acc.update — per-device-type async/wait shape mirroring `acc.parallel`,
  // plus an `ifPresent` UnitAttr (no `async` / `wait` keyword-only attrs;
  // those are encoded via `*Only` device-type arrays). Defining ops accepted
  // for `dataOperands`: `acc.update_device`, `acc.update_host`,
  // `acc.getdeviceptr`.
  func.func @update_minimal(%a : memref<f32>) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update dataOperands(%d : memref<f32>)
    func.return
  }
  // CHECK-LABEL: func.func @update_minimal(
  // CHECK:         acc.update dataOperands(%{{.*}} : memref<f32>)

  func.func @update_async_bare(%a : memref<f32>) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update async dataOperands(%d : memref<f32>)
    func.return
  }
  // CHECK-LABEL: func.func @update_async_bare(
  // CHECK:         acc.update async dataOperands(%{{.*}} : memref<f32>)

  func.func @update_async_operand(%a : memref<f32>, %v : i64) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update async(%v : i64) dataOperands(%d : memref<f32>)
    func.return
  }
  // CHECK-LABEL: func.func @update_async_operand(
  // CHECK:         acc.update async(%{{.*}} : i64) dataOperands(%{{.*}} : memref<f32>)

  func.func @update_wait_devnum(%a : memref<f32>, %dn : i64, %w : i32) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update wait({devnum: %dn : i64, %w : i32}) dataOperands(%d : memref<f32>)
    func.return
  }
  // CHECK-LABEL: func.func @update_wait_devnum(
  // CHECK:         acc.update wait({devnum: %{{.*}} : i64, %{{.*}} : i32}) dataOperands(%{{.*}} : memref<f32>)

  func.func @update_if_present(%a : memref<f32>, %c : i1) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update if(%c) dataOperands(%d : memref<f32>) attributes {ifPresent}
    func.return
  }
  // CHECK-LABEL: func.func @update_if_present(
  // CHECK:         acc.update if(%{{.*}}) dataOperands(%{{.*}} : memref<f32>) attributes {ifPresent}

  // Generic-form roundtrip insurance for `acc.update` — four-segment
  // operandSegmentSizes (if, async, wait, dataOperands).
  func.func @update_generic_roundtrip(%a : memref<f32>) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    "acc.update"(%d) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<f32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @update_generic_roundtrip(
  // CHECK:         acc.update dataOperands(%{{.*}} : memref<f32>)

  // acc.terminator is the value-less generic terminator. It currently only
  // permits `acc.kernels` as a parent (per HasParent on TerminatorOp); other
  // region ops in the OpenACC dialect (acc.data, acc.host_data, …) will be
  // appended to the parent tuple as they land.
  func.func @terminator_inside_kernels() {
    acc.kernels {
      acc.terminator
    }
    func.return
  }
  // CHECK-LABEL: func.func @terminator_inside_kernels() {
  // CHECK-NEXT:    acc.kernels {
  // CHECK-NEXT:      acc.terminator
  // CHECK-NEXT:    }

  // Generic-form roundtrip insurance for the value-less terminator.
  func.func @terminator_generic_roundtrip_retained() {
    acc.kernels {
      "acc.terminator"() : () -> ()
    }
    func.return
  }
  // CHECK-LABEL: func.func @terminator_generic_roundtrip_retained() {
  // CHECK-NEXT:    acc.kernels {
  // CHECK-NEXT:      acc.terminator
  // CHECK-NEXT:    }

  // acc.declare_enter — entry to an implicit declare data region. Yields
  // an `!acc.declare_token` that may be threaded into the matching
  // `acc.declare_exit`. Defining ops accepted for `dataOperands`: any of
  // copyin / copyout / create / deviceptr / getdeviceptr / present /
  // declare_device_resident / declare_link (per upstream's
  // `checkDeclareOperands` helper).
  func.func @declare_enter_basic(%a : memref<f32>) {
    %0 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32>
    %t = acc.declare_enter dataOperands(%0 : memref<f32>)
    acc.declare_exit token(%t) dataOperands(%0 : memref<f32>)
    func.return
  }
  // CHECK-LABEL: func.func @declare_enter_basic(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<f32>) -> memref<f32>
  // CHECK-NEXT:    %{{.*}} = acc.declare_enter dataOperands(%{{.*}} : memref<f32>)
  // CHECK-NEXT:    acc.declare_exit token(%{{.*}}) dataOperands(%{{.*}} : memref<f32>)

  // acc.declare_exit — when a `token` is present, `dataOperands` may be
  // empty (mirrors upstream's `requireAtLeastOneOperand=false` branch).
  func.func @declare_exit_token_only(%a : memref<f32>) {
    %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
    %t = acc.declare_enter dataOperands(%0 : memref<f32>)
    acc.declare_exit token(%t)
    func.return
  }
  // CHECK-LABEL: func.func @declare_exit_token_only(
  // CHECK:         acc.declare_exit token(%{{.*}})

  // acc.declare_exit — without a `token`, only `dataOperands` is given.
  // Defining ops accepted include `acc.getdeviceptr` for device-resident
  // tear-down (per upstream's example).
  func.func @declare_exit_no_token(%a : memref<f32>) {
    %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
    acc.declare_exit dataOperands(%0 : memref<f32>)
    func.return
  }
  // CHECK-LABEL: func.func @declare_exit_no_token(
  // CHECK:         acc.declare_exit dataOperands(%{{.*}} : memref<f32>)

  // acc.declare — structured declare region (no implicit terminator). The
  // body is just the implicit data region's lifetime; here it's empty.
  func.func @declare_region(%a : memref<f32>) {
    %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
    acc.declare dataOperands(%0 : memref<f32>) {
    }
    func.return
  }
  // CHECK-LABEL: func.func @declare_region(
  // CHECK:         acc.declare dataOperands(%{{.*}} : memref<f32>) {
  // CHECK-NEXT:    }

  // Generic-form roundtrip insurance for `acc.declare_enter` plus its
  // typed token result.
  func.func @declare_enter_generic_roundtrip_retained(%a : memref<f32>) {
    %0 = "acc.copyin"(%a) <{varType = f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (memref<f32>) -> memref<f32>
    %t = "acc.declare_enter"(%0) : (memref<f32>) -> !acc.declare_token
    "acc.declare_exit"(%t) <{operandSegmentSizes = array<i32: 1, 0>}> : (!acc.declare_token) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @declare_enter_generic_roundtrip_retained(
  // CHECK:         %{{.*}} = acc.declare_enter dataOperands(%{{.*}} : memref<f32>)
  // CHECK-NEXT:    acc.declare_exit token(%{{.*}})

  // Generic-form roundtrip insurance for `acc.declare_exit`'s
  // AttrSizedOperandSegments — the [0, 1] segments shape (no token,
  // single dataOperand).
  func.func @declare_exit_generic_roundtrip_retained(%a : memref<f32>) {
    %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
    "acc.declare_exit"(%0) <{operandSegmentSizes = array<i32: 0, 1>}> : (memref<f32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @declare_exit_generic_roundtrip_retained(
  // CHECK:         acc.declare_exit dataOperands(%{{.*}} : memref<f32>)

  // Generic-form roundtrip insurance for `acc.declare`.
  func.func @declare_generic_roundtrip_retained(%a : memref<f32>) {
    %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
    "acc.declare"(%0) ({
    }) : (memref<f32>) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @declare_generic_roundtrip_retained(
  // CHECK:         acc.declare dataOperands(%{{.*}} : memref<f32>) {
  // CHECK-NEXT:    }

  // ===========================================================================
  // acc.loop
  // ===========================================================================

  // Container-like loop (no induction variables) — the `independent` /
  // `seq` / `auto` device-`none` entry is required by the verifier.
  func.func @loop_empty() {
    acc.loop {
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    func.return
  }
  // CHECK-LABEL: func.func @loop_empty() {
  // CHECK-NEXT:    acc.loop {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {independent = [#acc.device_type<none>]}

  // Loop with induction variables — the `control(...) = (...) to (...)
  // step (...)` header is parsed/printed by the LoopControl directive.
  func.func @loop_control(%lb : index, %ub : index, %st : index) {
    acc.loop control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_control(
  // CHECK:         acc.loop control(%{{.*}} : index) = (%{{.*}} : index) to (%{{.*}} : index) step (%{{.*}} : index) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

  // Bare gang/worker/vector keywords land as device-`none` arrays.
  func.func @loop_bare_par_keywords(%lb : index, %ub : index, %st : index) {
    acc.loop gang worker vector control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_bare_par_keywords(
  // CHECK:         acc.loop gang worker vector control(%{{.*}} : index) = (%{{.*}} : index) to (%{{.*}} : index) step (%{{.*}} : index) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

  // gang clause with `num=` / `static=` / `dim=` operand groups.
  func.func @loop_gang_operands(%v : i64, %lb : index, %ub : index, %st : index) {
    acc.loop gang({num=%v : i64, static=%v : i64}) worker(%v : i64) vector(%v : i64) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_gang_operands(
  // CHECK:         acc.loop gang({num=%{{.*}} : i64, static=%{{.*}} : i64}) worker(%{{.*}} : i64) vector(%{{.*}} : i64) control(%{{.*}} : index)

  // tile clause with brace-grouped operands.
  func.func @loop_tile(%v : i64, %lb : index, %ub : index, %st : index) {
    acc.loop tile({%v : i64, %v : i64}) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_tile(
  // CHECK:         acc.loop tile({%{{.*}} : i64, %{{.*}} : i64}) control(%{{.*}} : index)

  // Combined construct keyword — `combined(parallel)` decomposes a
  // `parallel loop` directive.
  func.func @loop_combined(%lb : index, %ub : index, %st : index) {
    acc.loop combined(parallel) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_combined(
  // CHECK:         acc.loop combined(parallel) control(%{{.*}} : index)

  // private / firstprivate / reduction operand groups.
  func.func @loop_data_clauses(%a : memref<10xf32>, %lb : index, %ub : index, %st : index) {
    %p = acc.private varPtr(%a : memref<10xf32>) -> memref<10xf32>
    %f = acc.firstprivate varPtr(%a : memref<10xf32>) -> memref<10xf32>
    %r = acc.reduction varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.loop private(%p : memref<10xf32>) firstprivate(%f : memref<10xf32>) reduction(%r : memref<10xf32>) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_data_clauses(
  // CHECK:         acc.loop private(%{{.*}} : memref<10xf32>) firstprivate(%{{.*}} : memref<10xf32>) reduction(%{{.*}} : memref<10xf32>) control(%{{.*}} : index)

  // cache operand list.
  func.func @loop_cache(%a : memref<10xf32>, %lb : index, %ub : index, %st : index) {
    %b = acc.cache varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.loop cache(%b : memref<10xf32>) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_cache(
  // CHECK:         acc.loop cache(%{{.*}} : memref<10xf32>) control(%{{.*}} : index)

  // Generic-form roundtrip insurance for `acc.loop` — proves the bare
  // generic surface (operandSegmentSizes + properties) still parses.
  func.func @loop_generic_roundtrip_retained(%lb : index, %ub : index, %st : index) {
    "acc.loop"(%lb, %ub, %st) <{independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
    ^bb0(%iv: index):
      "acc.yield"() : () -> ()
    }) : (index, index, index) -> ()
    func.return
  }
  // CHECK-LABEL: func.func @loop_generic_roundtrip_retained(
  // CHECK:         acc.loop control(%{{.*}} : index) = (%{{.*}} : index) to (%{{.*}} : index) step (%{{.*}} : index) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

  // gang clause with a keyword-only `[#dt]` device-type list (no operands).
  func.func @loop_gang_kw_only_dts(%lb : index, %ub : index, %st : index) {
    acc.loop gang([#acc.device_type<nvidia>]) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_gang_kw_only_dts(
  // CHECK:         acc.loop gang([#acc.device_type<nvidia>]) control(%{{.*}} : index)

  // gang clause mixing keyword-only DT list and operand groups.
  func.func @loop_gang_mixed(%v : i64, %lb : index, %ub : index, %st : index) {
    acc.loop gang([#acc.device_type<nvidia>], {num=%v : i64} [#acc.device_type<default>]) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_gang_mixed(
  // CHECK:         acc.loop gang([#acc.device_type<nvidia>], {num=%{{.*}} : i64} [#acc.device_type<default>]) control(%{{.*}} : index)

  // Multiple gang operand groups — exercises the `, ` separator between
  // groups in GangClause's printer.
  func.func @loop_gang_multi_group(%v : i64, %lb : index, %ub : index, %st : index) {
    acc.loop gang({num=%v : i64} [#acc.device_type<nvidia>], {dim=%v : i64} [#acc.device_type<default>]) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
      acc.yield
    } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
    func.return
  }
  // CHECK-LABEL: func.func @loop_gang_multi_group(
  // CHECK:         acc.loop gang({num=%{{.*}} : i64} [#acc.device_type<nvidia>], {dim=%{{.*}} : i64} [#acc.device_type<default>]) control(%{{.*}} : index)

  // collapse with matching counts (1 entry each) — exercises the verifier
  // path past the collapse-count check on the happy path.
  func.func @loop_collapse() {
    acc.loop {
      acc.yield
    } attributes {collapse = [2 : i64], collapseDeviceType = [#acc.device_type<none>], independent = [#acc.device_type<none>]}
    func.return
  }
  // CHECK-LABEL: func.func @loop_collapse() {
  // CHECK:         acc.loop {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {collapse = [2 : i64], collapseDeviceType = [#acc.device_type<none>], independent = [#acc.device_type<none>]}

  // seq + gang with non-overlapping device types — exercises the
  // gang/worker/vector incompatibility check's no-conflict branch. The
  // round-trip prints `gang([#nvidia])` via the GangClause directive's
  // keyword-only DT spelling (since `gang`'s sole entry isn't `#none`).
  func.func @loop_seq_gang_disjoint() {
    acc.loop {
      acc.yield
    } attributes {seq = [#acc.device_type<none>], gang = [#acc.device_type<nvidia>]}
    func.return
  }
  // CHECK-LABEL: func.func @loop_seq_gang_disjoint() {
  // CHECK:         acc.loop gang([#acc.device_type<nvidia>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {seq = [#acc.device_type<none>]}
}
