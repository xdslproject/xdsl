// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

builtin.module {
  func.func @empty() {
    acc.parallel {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @empty() {
  // CHECK-NEXT:    acc.parallel {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @self_if(%c : i1) {
    acc.parallel self(%c) if(%c) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @self_if(
  // CHECK:         acc.parallel self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @combined_default_self() {
    acc.parallel combined(loop) {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK:       func.func @combined_default_self() {
  // CHECK-NEXT:    acc.parallel combined(loop) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  func.func @async_bare() {
    acc.parallel async {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @async_bare() {
  // CHECK-NEXT:    acc.parallel async {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @async_one_operand(%a : i64) {
    acc.parallel async(%a : i64) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @async_one_operand(
  // CHECK:         acc.parallel async(%{{.*}} : i64) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @async_operand_with_dt(%a : i64) {
    acc.parallel async(%a : i64 [#acc.device_type<default>]) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @async_operand_with_dt(
  // CHECK:         acc.parallel async(%{{.*}} : i64 [#acc.device_type<default>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @num_gangs_single(%a : i32) {
    acc.parallel num_gangs({%a : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @num_gangs_single(
  // CHECK:         acc.parallel num_gangs({%{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @num_gangs_multi(%a : i32, %b : i32, %c : index) {
    acc.parallel num_gangs({%a : i32, %b : i32, %c : index}) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @num_gangs_multi(
  // CHECK:         acc.parallel num_gangs({%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : index}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @num_gangs_multi_dt(%a : i32, %b : i32, %c : i32) {
    acc.parallel num_gangs({%a : i32} [#acc.device_type<default>], {%b : i32, %c : i32} [#acc.device_type<nvidia>]) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @num_gangs_multi_dt(
  // CHECK:         acc.parallel num_gangs({%{{.*}} : i32} [#acc.device_type<default>], {%{{.*}} : i32, %{{.*}} : i32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @num_workers_vector_length(%a : i64, %b : i32, %c : i32) {
    acc.parallel num_workers(%a : i64 [#acc.device_type<default>], %b : i32 [#acc.device_type<nvidia>]) vector_length(%c : i32) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @num_workers_vector_length(
  // CHECK:         acc.parallel num_workers(%{{.*}} : i64 [#acc.device_type<default>], %{{.*}} : i32 [#acc.device_type<nvidia>]) vector_length(%{{.*}} : i32) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_bare() {
    acc.parallel wait {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @wait_bare() {
  // CHECK-NEXT:    acc.parallel wait {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_group(%a : i64, %b : i32) {
    acc.parallel wait({%a : i64, %b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @wait_group(
  // CHECK:         acc.parallel wait({%{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_devnum(%a : i64, %b : i32) {
    acc.parallel wait({devnum: %a : i64, %b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @wait_devnum(
  // CHECK:         acc.parallel wait({devnum: %{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @wait_mixed(%a : i64, %b : i32) {
    acc.parallel wait([#acc.device_type<nvidia>], {devnum: %a : i64, %b : i32} [#acc.device_type<default>]) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @wait_mixed(
  // CHECK:         acc.parallel wait([#acc.device_type<nvidia>], {devnum: %{{.*}} : i64, %{{.*}} : i32} [#acc.device_type<default>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @test_entire(%c : i1, %a : i32, %b : i64) {
    acc.parallel combined(loop) async(%b : i64) num_workers(%b : i64) vector_length(%a : i32) wait({%b : i64}) self(%c) if(%c) {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK:       func.func @test_entire(
  // CHECK:         acc.parallel combined(loop) async(%{{.*}} : i64) num_workers(%{{.*}} : i64) vector_length(%{{.*}} : i32) wait({%{{.*}} : i64}) self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  func.func @serial_empty() {
    acc.serial {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_empty() {
  // CHECK-NEXT:    acc.serial {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_self_if(%c : i1) {
    acc.serial self(%c) if(%c) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_self_if(
  // CHECK:         acc.serial self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_combined_default_self() {
    acc.serial combined(loop) {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK:       func.func @serial_combined_default_self() {
  // CHECK-NEXT:    acc.serial combined(loop) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  func.func @serial_async_bare() {
    acc.serial async {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_async_bare() {
  // CHECK-NEXT:    acc.serial async {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_async_one_operand(%a : i64) {
    acc.serial async(%a : i64) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_async_one_operand(
  // CHECK:         acc.serial async(%{{.*}} : i64) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_async_operand_with_dt(%a : i64) {
    acc.serial async(%a : i64 [#acc.device_type<default>]) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_async_operand_with_dt(
  // CHECK:         acc.serial async(%{{.*}} : i64 [#acc.device_type<default>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_wait_bare() {
    acc.serial wait {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_wait_bare() {
  // CHECK-NEXT:    acc.serial wait {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_wait_group(%a : i64, %b : i32) {
    acc.serial wait({%a : i64, %b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_wait_group(
  // CHECK:         acc.serial wait({%{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_wait_devnum(%a : i64, %b : i32) {
    acc.serial wait({devnum: %a : i64, %b : i32}) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_wait_devnum(
  // CHECK:         acc.serial wait({devnum: %{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_wait_mixed(%a : i64, %b : i32) {
    acc.serial wait([#acc.device_type<nvidia>], {devnum: %a : i64, %b : i32} [#acc.device_type<default>]) {
      acc.yield
    }
    func.return
  }
  // CHECK:       func.func @serial_wait_mixed(
  // CHECK:         acc.serial wait([#acc.device_type<nvidia>], {devnum: %{{.*}} : i64, %{{.*}} : i32} [#acc.device_type<default>]) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    }

  func.func @serial_test_entire(%c : i1, %b : i64) {
    acc.serial combined(loop) async(%b : i64) wait({%b : i64}) self(%c) if(%c) {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK:       func.func @serial_test_entire(
  // CHECK:         acc.serial combined(loop) async(%{{.*}} : i64) wait({%{{.*}} : i64}) self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:      acc.yield
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  // acc.kernels models upstream's `AnyRegion` body (NoTerminator); upstream
  // mlir-opt rejects acc.yield inside acc.kernels (yield's ParentOneOf list
  // excludes KernelsOp), so all bodies here are empty until acc.terminator
  // is introduced later in the roadmap.
  func.func @kernels_empty() {
    acc.kernels {
    }
    func.return
  }
  // CHECK:       func.func @kernels_empty() {
  // CHECK-NEXT:    acc.kernels {
  // CHECK-NEXT:    }

  func.func @kernels_self_if(%c : i1) {
    acc.kernels self(%c) if(%c) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_self_if(
  // CHECK:         acc.kernels self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:    }

  func.func @kernels_combined_default_self() {
    acc.kernels combined(loop) {
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK:       func.func @kernels_combined_default_self() {
  // CHECK-NEXT:    acc.kernels combined(loop) {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  func.func @kernels_async_bare() {
    acc.kernels async {
    }
    func.return
  }
  // CHECK:       func.func @kernels_async_bare() {
  // CHECK-NEXT:    acc.kernels async {
  // CHECK-NEXT:    }

  func.func @kernels_async_one_operand(%a : i64) {
    acc.kernels async(%a : i64) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_async_one_operand(
  // CHECK:         acc.kernels async(%{{.*}} : i64) {
  // CHECK-NEXT:    }

  func.func @kernels_async_operand_with_dt(%a : i64) {
    acc.kernels async(%a : i64 [#acc.device_type<default>]) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_async_operand_with_dt(
  // CHECK:         acc.kernels async(%{{.*}} : i64 [#acc.device_type<default>]) {
  // CHECK-NEXT:    }

  func.func @kernels_num_gangs_single(%a : i32) {
    acc.kernels num_gangs({%a : i32}) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_num_gangs_single(
  // CHECK:         acc.kernels num_gangs({%{{.*}} : i32}) {
  // CHECK-NEXT:    }

  func.func @kernels_num_gangs_multi_dt(%a : i32, %b : i32, %c : i32) {
    acc.kernels num_gangs({%a : i32} [#acc.device_type<default>], {%b : i32, %c : i32} [#acc.device_type<nvidia>]) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_num_gangs_multi_dt(
  // CHECK:         acc.kernels num_gangs({%{{.*}} : i32} [#acc.device_type<default>], {%{{.*}} : i32, %{{.*}} : i32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT:    }

  func.func @kernels_num_workers_vector_length(%a : i64, %b : i32, %c : i32) {
    acc.kernels num_workers(%a : i64 [#acc.device_type<default>], %b : i32 [#acc.device_type<nvidia>]) vector_length(%c : i32) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_num_workers_vector_length(
  // CHECK:         acc.kernels num_workers(%{{.*}} : i64 [#acc.device_type<default>], %{{.*}} : i32 [#acc.device_type<nvidia>]) vector_length(%{{.*}} : i32) {
  // CHECK-NEXT:    }

  func.func @kernels_wait_bare() {
    acc.kernels wait {
    }
    func.return
  }
  // CHECK:       func.func @kernels_wait_bare() {
  // CHECK-NEXT:    acc.kernels wait {
  // CHECK-NEXT:    }

  func.func @kernels_wait_group(%a : i64, %b : i32) {
    acc.kernels wait({%a : i64, %b : i32}) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_wait_group(
  // CHECK:         acc.kernels wait({%{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:    }

  func.func @kernels_wait_devnum(%a : i64, %b : i32) {
    acc.kernels wait({devnum: %a : i64, %b : i32}) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_wait_devnum(
  // CHECK:         acc.kernels wait({devnum: %{{.*}} : i64, %{{.*}} : i32}) {
  // CHECK-NEXT:    }

  func.func @kernels_wait_mixed(%a : i64, %b : i32) {
    acc.kernels wait([#acc.device_type<nvidia>], {devnum: %a : i64, %b : i32} [#acc.device_type<default>]) {
    }
    func.return
  }
  // CHECK:       func.func @kernels_wait_mixed(
  // CHECK:         acc.kernels wait([#acc.device_type<nvidia>], {devnum: %{{.*}} : i64, %{{.*}} : i32} [#acc.device_type<default>]) {
  // CHECK-NEXT:    }

  func.func @kernels_test_entire(%c : i1, %a : i32, %b : i64) {
    acc.kernels combined(loop) async(%b : i64) num_workers(%b : i64) vector_length(%a : i32) wait({%b : i64}) self(%c) if(%c) {
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  // CHECK:       func.func @kernels_test_entire(
  // CHECK:         acc.kernels combined(loop) async(%{{.*}} : i64) num_workers(%{{.*}} : i64) vector_length(%{{.*}} : i32) wait({%{{.*}} : i64}) self(%{{.*}}) if(%{{.*}}) {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}

  // The trailing `strideInBytes = false` attribute is omitted by mlir-opt's
  // pretty printer (it matches the default) but appears explicitly in the
  // generic roundtrip path because mlir-opt prints the property in generic
  // form. CHECK lines below match the prefix only so both pipelines pass.
  func.func @bounds_lb_ub_stride(%c0 : index, %c9 : index, %c1 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index) stride(%c1 : index)
    func.return
  }
  // CHECK:       func.func @bounds_lb_ub_stride(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index)

  func.func @bounds_extent_only(%c9 : index) {
    %b = acc.bounds extent(%c9 : index)
    func.return
  }
  // CHECK:       func.func @bounds_extent_only(
  // CHECK:         %{{.*}} = acc.bounds extent(%{{.*}} : index)

  func.func @bounds_full(%c1 : index, %c20 : index, %c4 : index) {
    %b = acc.bounds lowerbound(%c1 : index) upperbound(%c20 : index) extent(%c20 : index) stride(%c4 : index) startIdx(%c1 : index) {strideInBytes = true}
    func.return
  }
  // CHECK:       func.func @bounds_full(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index) {strideInBytes = true}

  func.func @bounds_accessors(%b : !acc.data_bounds_ty) {
    %lb = acc.get_lowerbound %b : (!acc.data_bounds_ty) -> index
    %ub = acc.get_upperbound %b : (!acc.data_bounds_ty) -> index
    %stride = acc.get_stride %b : (!acc.data_bounds_ty) -> index
    %extent = acc.get_extent %b : (!acc.data_bounds_ty) -> index
    func.return
  }
  // CHECK:       func.func @bounds_accessors(
  // CHECK:         %{{.*}} = acc.get_lowerbound %{{.*}} : (!acc.data_bounds_ty) -> index
  // CHECK-NEXT:    %{{.*}} = acc.get_upperbound %{{.*}} : (!acc.data_bounds_ty) -> index
  // CHECK-NEXT:    %{{.*}} = acc.get_stride %{{.*}} : (!acc.data_bounds_ty) -> index
  // CHECK-NEXT:    %{{.*}} = acc.get_extent %{{.*}} : (!acc.data_bounds_ty) -> index

  // Entry data-clause ops (PR 6b: copyin / create / present). Each MLIR
  // interop entry round-trips through both the pretty and generic surfaces;
  // covers `acc.copyin` exhaustively and the other two minimally.
  func.func @copyin_minimal(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_minimal(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @copyin_with_var_type(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_with_var_type(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) varType(tensor<10xf32>) -> memref<10xf32>

  func.func @copyin_with_var_ptr_ptr(%a : memref<10xf32>, %p : memref<memref<10xf32>>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) varPtrPtr(%p : memref<memref<10xf32>>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_with_var_ptr_ptr(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) varPtrPtr(%{{.*}} : memref<memref<10xf32>>) -> memref<10xf32>

  func.func @copyin_with_bounds(%a : memref<10xf32>, %c0 : index, %c9 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
    %r = acc.copyin varPtr(%a : memref<10xf32>) bounds(%b) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_with_bounds(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index)
  // CHECK-NEXT:    %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) bounds(%{{.*}}) -> memref<10xf32>

  func.func @copyin_async_bare(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) async -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_async_bare(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) async -> memref<10xf32>

  func.func @copyin_async_operand(%a : memref<10xf32>, %async : i32) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) async(%async : i32) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_async_operand(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) async(%{{.*}} : i32) -> memref<10xf32>

  func.func @copyin_async_operand_dt(%a : memref<10xf32>, %async : i32) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) async(%async : i32 [#acc.device_type<nvidia>]) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_async_operand_dt(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) async(%{{.*}} : i32 [#acc.device_type<nvidia>]) -> memref<10xf32>

  func.func @copyin_clause_override(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyin_readonly>, modifiers = #acc<data_clause_modifier readonly>, name = "myvar"}
    func.return
  }
  // CHECK:       func.func @copyin_clause_override(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyin_readonly>, modifiers = #acc<data_clause_modifier readonly>, name = "myvar"}

  func.func @create_minimal(%a : memref<10xf32>) {
    %r = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @create_minimal(
  // CHECK:         %{{.*}} = acc.create varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @create_with_copyout_clause(%a : memref<10xf32>) {
    %r = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout>}
    func.return
  }
  // CHECK:       func.func @create_with_copyout_clause(
  // CHECK:         %{{.*}} = acc.create varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout>}

  func.func @present_minimal(%a : memref<10xf32>) {
    %r = acc.present varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @present_minimal(
  // CHECK:         %{{.*}} = acc.present varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
}
