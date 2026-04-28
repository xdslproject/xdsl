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

  // acc.kernels uses the upstream `NoTerminator` body shape: the body can
  // be empty or contain ops that lower to a kernels region. acc.yield is
  // *not* a valid terminator inside acc.kernels (upstream's acc.yield
  // ParentOneOf list excludes KernelsOp); the dedicated acc.terminator op
  // lands later in the roadmap.
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
}
