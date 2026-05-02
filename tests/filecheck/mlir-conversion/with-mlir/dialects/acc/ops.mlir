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

  // acc.kernels uses `SingleBlockImplicitTerminator(TerminatorOp)` on both
  // sides: the pretty parser auto-inserts `acc.terminator`, the printer
  // elides it. So `acc.kernels { }` round-trips identically through xDSL
  // and mlir-opt even though the in-memory IR has the terminator present.
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

  // acc.data — round-trips through mlir-opt's OpenACC dialect in both
  // pretty and generic form. defaultAttr is required on bodies with no
  // operand to satisfy upstream's verifier (matches xDSL's port).
  func.func @data_default_attr() {
    acc.data {
    } attributes {defaultAttr = #acc<defaultvalue none>}
    func.return
  }
  // CHECK:       func.func @data_default_attr() {
  // CHECK-NEXT:    acc.data {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue none>}

  func.func @data_data_operand(%a : memref<10xf32>) {
    %p = acc.present varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.data dataOperands(%p : memref<10xf32>) {
    }
    func.return
  }
  // CHECK:       func.func @data_data_operand(
  // CHECK:         acc.data dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    }

  func.func @data_if_async_wait(%c : i1, %a : i64, %w : i64) {
    acc.data if(%c) async(%a : i64) wait({%w : i64}) {
    } attributes {defaultAttr = #acc<defaultvalue none>}
    func.return
  }
  // CHECK:       func.func @data_if_async_wait(
  // CHECK:         acc.data if(%{{.*}}) async(%{{.*}} : i64) wait({%{{.*}} : i64}) {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue none>}

  func.func @data_async_bare() {
    acc.data async {
    } attributes {defaultAttr = #acc<defaultvalue none>}
    func.return
  }
  // CHECK:       func.func @data_async_bare() {
  // CHECK-NEXT:    acc.data async {
  // CHECK-NEXT:    } attributes {defaultAttr = #acc<defaultvalue none>}

  // acc.host_data — operands must be defined by acc.use_device, and the
  // op carries an `ifPresent` UnitAttr instead of a default clause.
  func.func @host_data_minimal(%a : memref<10xf32>) {
    %u = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.host_data dataOperands(%u : memref<10xf32>) {
    }
    func.return
  }
  // CHECK:       func.func @host_data_minimal(
  // CHECK:         acc.host_data dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    }

  func.func @host_data_if_present(%a : memref<10xf32>) {
    %u = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.host_data dataOperands(%u : memref<10xf32>) {
    } attributes {ifPresent}
    func.return
  }
  // CHECK:       func.func @host_data_if_present(
  // CHECK:         acc.host_data dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    } attributes {ifPresent}

  func.func @host_data_if_cond(%c : i1, %a : memref<10xf32>) {
    %u = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.host_data if(%c) dataOperands(%u : memref<10xf32>) {
    }
    func.return
  }
  // CHECK:       func.func @host_data_if_cond(
  // CHECK:         acc.host_data if(%{{.*}}) dataOperands(%{{.*}} : memref<10xf32>) {
  // CHECK-NEXT:    }

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

  // Entry data-clause ops — `copyin` / `create` / `present`. Each MLIR
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

  // Remaining seven entry data-clause leaves. Each gets one MLIR interop
  // entry — proves xDSL emission is bit-compatible with upstream and the
  // per-op `dataClause` default round-trips through MLIR.
  func.func @nocreate_minimal(%a : memref<10xf32>) {
    %r = acc.nocreate varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @nocreate_minimal(
  // CHECK:         %{{.*}} = acc.nocreate varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @attach_minimal(%a : memref<10xf32>) {
    %r = acc.attach varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @attach_minimal(
  // CHECK:         %{{.*}} = acc.attach varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @deviceptr_minimal(%a : memref<10xf32>) {
    %r = acc.deviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @deviceptr_minimal(
  // CHECK:         %{{.*}} = acc.deviceptr varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @use_device_minimal(%a : memref<10xf32>) {
    %r = acc.use_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @use_device_minimal(
  // CHECK:         %{{.*}} = acc.use_device varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @cache_minimal(%a : memref<10xf32>) {
    %r = acc.cache varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @cache_minimal(
  // CHECK:         %{{.*}} = acc.cache varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @declare_device_resident_minimal(%a : memref<10xf32>) {
    %r = acc.declare_device_resident varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @declare_device_resident_minimal(
  // CHECK:         %{{.*}} = acc.declare_device_resident varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @declare_link_minimal(%a : memref<10xf32>) {
    %r = acc.declare_link varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @declare_link_minimal(
  // CHECK:         %{{.*}} = acc.declare_link varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @copyout_minimal(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.copyout accPtr(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @copyout_minimal(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @copyout_with_var_type(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.copyout accPtr(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>) varType(tensor<10xf32>)
    func.return
  }
  // CHECK:       func.func @copyout_with_var_type(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>) varType(tensor<10xf32>)

  func.func @copyout_with_bounds(%d : memref<10xf32>, %h : memref<10xf32>, %c0 : index, %c9 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
    acc.copyout accPtr(%d : memref<10xf32>) bounds(%b) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @copyout_with_bounds(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index)
  // CHECK-NEXT:    acc.copyout accPtr(%{{.*}} : memref<10xf32>) bounds(%{{.*}}) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @copyout_async_operand_dt(%d : memref<10xf32>, %h : memref<10xf32>, %async : i32) {
    acc.copyout accPtr(%d : memref<10xf32>) async(%async : i32 [#acc.device_type<nvidia>]) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @copyout_async_operand_dt(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) async(%{{.*}} : i32 [#acc.device_type<nvidia>]) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @copyout_clause_override(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.copyout accPtr(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>) {dataClause = #acc<data_clause acc_copyout_zero>, modifiers = #acc<data_clause_modifier zero>, name = "myvar"}
    func.return
  }
  // CHECK:       func.func @copyout_clause_override(
  // CHECK:         acc.copyout accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>) {dataClause = #acc<data_clause acc_copyout_zero>, modifiers = #acc<data_clause_modifier zero>, name = "myvar"}

  func.func @update_host_minimal(%d : memref<10xf32>, %h : memref<10xf32>) {
    acc.update_host accPtr(%d : memref<10xf32>) to varPtr(%h : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @update_host_minimal(
  // CHECK:         acc.update_host accPtr(%{{.*}} : memref<10xf32>) to varPtr(%{{.*}} : memref<10xf32>)

  func.func @getdeviceptr_minimal(%a : memref<10xf32>) {
    %r = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @getdeviceptr_minimal(
  // CHECK:         %{{.*}} = acc.getdeviceptr varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @update_device_minimal(%a : memref<10xf32>) {
    %r = acc.update_device varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @update_device_minimal(
  // CHECK:         %{{.*}} = acc.update_device varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @delete_minimal(%d : memref<10xf32>) {
    acc.delete accPtr(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @delete_minimal(
  // CHECK:         acc.delete accPtr(%{{.*}} : memref<10xf32>)

  func.func @delete_with_bounds_async(%d : memref<10xf32>, %async : i32, %c0 : index, %c9 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
    acc.delete accPtr(%d : memref<10xf32>) bounds(%b) async(%async : i32)
    func.return
  }
  // CHECK:       func.func @delete_with_bounds_async(
  // CHECK:         %{{.*}} = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index)
  // CHECK-NEXT:    acc.delete accPtr(%{{.*}} : memref<10xf32>) bounds(%{{.*}}) async(%{{.*}} : i32)

  func.func @delete_clause_override(%d : memref<10xf32>) {
    acc.delete accPtr(%d : memref<10xf32>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier alwaysout>, name = "myvar"}
    func.return
  }
  // CHECK:       func.func @delete_clause_override(
  // CHECK:         acc.delete accPtr(%{{.*}} : memref<10xf32>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier alwaysout>, name = "myvar"}

  func.func @detach_minimal(%d : memref<10xf32>) {
    acc.detach accPtr(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @detach_minimal(
  // CHECK:         acc.detach accPtr(%{{.*}} : memref<10xf32>)

  // `recipe(@sym)` interop: upstream's pretty-form spelling for the
  // `recipe` SymbolRefAttr property. The `DataEntryOilist` directive's
  // anchor on `$recipe` must produce the inline form that mlir-opt
  // re-emits identically, otherwise the round-trip diverges.
  func.func @copyin_with_recipe(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) recipe(@some_recipe) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_with_recipe(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) recipe(@some_recipe) -> memref<10xf32>

  // Attr-dict spelling for `recipe` interop: xDSL's `ParsePropInAttrDict`
  // fallback parses `{recipe = @sym}` and emits the inline `recipe(@sym)`
  // form. mlir-opt re-emits the same inline form, so the round-trip lands
  // bit-identical to the canonical pretty form.
  func.func @copyin_recipe_attr_dict_input(%a : memref<10xf32>) {
    %r = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {recipe = @some_recipe}
    func.return
  }
  // CHECK:       func.func @copyin_recipe_attr_dict_input(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) recipe(@some_recipe) -> memref<10xf32>

  // Order-free input interop: clauses written in non-canonical order on
  // input get normalized to canonical td-definition order
  // (varPtrPtr → bounds → async → recipe) by both xDSL and mlir-opt.
  // Round-trips bit-identical even though the source spelling differs.
  func.func @copyin_clauses_reordered(%a : memref<10xf32>, %p : memref<memref<10xf32>>, %c0 : index, %c9 : index) {
    %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
    %r = acc.copyin varPtr(%a : memref<10xf32>) recipe(@some_recipe) async bounds(%b) varPtrPtr(%p : memref<memref<10xf32>>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @copyin_clauses_reordered(
  // CHECK:         %{{.*}} = acc.copyin varPtr(%{{.*}} : memref<10xf32>) varPtrPtr(%{{.*}} : memref<memref<10xf32>>) bounds(%{{.*}}) async recipe(@some_recipe) -> memref<10xf32>

  // Privatization data-clause ops. Same `_DataEntryOperation` shape as
  // `acc.copyin` / `acc.create` etc., differing only in per-op `dataClause`
  // default and op name. Upstream's verifier requires a `recipe` attribute
  // on `acc.private` / `acc.firstprivate` / `acc.reduction` (recipe is
  // mandatory whenever the op decomposes a privatization / reduction
  // clause). `acc.firstprivate_map` does *not* require a recipe upstream,
  // so its minimal form is exercised here too. Recipe-less variants for
  // the other three live in the xDSL-only test.
  func.func @private_with_recipe(%a : memref<10xf32>) {
    %r = acc.private varPtr(%a : memref<10xf32>) recipe(@priv_min) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @private_with_recipe(
  // CHECK:         %{{.*}} = acc.private varPtr(%{{.*}} : memref<10xf32>) recipe(@priv_min) -> memref<10xf32>

  func.func @firstprivate_with_recipe(%a : memref<10xf32>) {
    %r = acc.firstprivate varPtr(%a : memref<10xf32>) recipe(@fp_min) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @firstprivate_with_recipe(
  // CHECK:         %{{.*}} = acc.firstprivate varPtr(%{{.*}} : memref<10xf32>) recipe(@fp_min) -> memref<10xf32>

  func.func @firstprivate_map_minimal(%a : memref<10xf32>) {
    %r = acc.firstprivate_map varPtr(%a : memref<10xf32>) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @firstprivate_map_minimal(
  // CHECK:         %{{.*}} = acc.firstprivate_map varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>

  func.func @reduction_with_recipe(%a : memref<10xf32>) {
    %r = acc.reduction varPtr(%a : memref<10xf32>) recipe(@red_add_i64) -> memref<10xf32>
    func.return
  }
  // CHECK:       func.func @reduction_with_recipe(
  // CHECK:         %{{.*}} = acc.reduction varPtr(%{{.*}} : memref<10xf32>) recipe(@red_add_i64) -> memref<10xf32>

  // Per-op `dataClause` default agreement: feeding each leaf's expected
  // default explicitly through the attr-dict and asserting that *both*
  // xDSL and mlir-opt elide it on print proves the two sides agree on
  // each leaf's default value. A wrong default at either end would
  // surface here as `dataClause` no longer eliding through the
  // round-trip. Recipe attached on the three ops where upstream's
  // verifier requires it; `firstprivate_map` carries no recipe (upstream
  // does not require one).
  func.func @private_dataclause_default_elided(%a : memref<10xf32>) {
    %r = acc.private varPtr(%a : memref<10xf32>) recipe(@priv_min) -> memref<10xf32> {dataClause = #acc<data_clause acc_private>}
    func.return
  }
  // CHECK:       func.func @private_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.private varPtr(%{{.*}} : memref<10xf32>) recipe(@priv_min) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @firstprivate_dataclause_default_elided(%a : memref<10xf32>) {
    %r = acc.firstprivate varPtr(%a : memref<10xf32>) recipe(@fp_min) -> memref<10xf32> {dataClause = #acc<data_clause acc_firstprivate>}
    func.return
  }
  // CHECK:       func.func @firstprivate_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.firstprivate varPtr(%{{.*}} : memref<10xf32>) recipe(@fp_min) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  // `acc.firstprivate_map` shares the `acc_firstprivate` user-level
  // clause with `acc.firstprivate`, so its default is also
  // `acc_firstprivate`. Load-bearing under interop — if either xDSL or
  // mlir-opt thought the default was the op-name-derived value, the
  // elision would fail.
  func.func @firstprivate_map_dataclause_default_elided(%a : memref<10xf32>) {
    %r = acc.firstprivate_map varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_firstprivate>}
    func.return
  }
  // CHECK:       func.func @firstprivate_map_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.firstprivate_map varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  func.func @reduction_dataclause_default_elided(%a : memref<10xf32>) {
    %r = acc.reduction varPtr(%a : memref<10xf32>) recipe(@red_add_i64) -> memref<10xf32> {dataClause = #acc<data_clause acc_reduction>}
    func.return
  }
  // CHECK:       func.func @reduction_dataclause_default_elided(
  // CHECK:         %{{.*}} = acc.reduction varPtr(%{{.*}} : memref<10xf32>) recipe(@red_add_i64) -> memref<10xf32>
  // CHECK-NOT:     dataClause

  // Privatization recipes — top-level Symbol ops. Pretty form is
  // `@sym : type init { ... } [destroy { ... }]?` (and `copy {...}` between
  // for firstprivate). Both MLIR_ROUNDTRIP and MLIR_GENERIC_ROUNDTRIP must
  // pass — the latter exercises the `<{sym_name = ..., type = ...}>` props
  // dict that any consumer (e.g. `acc.private` referencing this recipe by
  // SymbolRefAttr) will need to round-trip.
  acc.private.recipe @priv_min : memref<10xf32> init {
  ^bb0(%arg0: memref<10xf32>):
    acc.yield %arg0 : memref<10xf32>
  }
  // CHECK:       acc.private.recipe @priv_min : memref<10xf32> init {
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
  // CHECK:       acc.private.recipe @priv_with_destroy : memref<10xf32> init {
  // CHECK:         } destroy {
  // CHECK:           acc.yield
  // CHECK-NEXT:    }

  acc.firstprivate.recipe @fp_min : memref<10xf32> init {
  ^bb0(%arg0: memref<10xf32>):
    acc.yield %arg0 : memref<10xf32>
  } copy {
  ^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
    acc.yield
  }
  // CHECK:       acc.firstprivate.recipe @fp_min : memref<10xf32> init {
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
  // CHECK:       acc.firstprivate.recipe @fp_with_destroy : memref<10xf32> init {
  // CHECK:         } copy {
  // CHECK:         } destroy {

  // Reduction recipe — load-bearing for MLIR interop because xDSL's pretty
  // form spells the operator inline as `<add>` (the
  // `ReductionOpKindAttr.print_parameter` output) which is exactly what
  // upstream's `assemblyFormat = "`<` $value `>`"` expects after the
  // `reduction_operator` keyword. Both `MLIR_ROUNDTRIP` and
  // `MLIR_GENERIC_ROUNDTRIP` must pass — the latter exercises the
  // `reductionOperator = #acc.reduction_operator<add>` opaque spelling
  // in the properties dict.
  acc.reduction.recipe @red_add_i64 : i64 reduction_operator <add> init {
  ^bb0(%arg0: i64):
    %c0 = arith.constant 0 : i64
    acc.yield %c0 : i64
  } combiner {
  ^bb0(%arg0: i64, %arg1: i64):
    %r = arith.addi %arg0, %arg1 : i64
    acc.yield %r : i64
  }
  // CHECK:       acc.reduction.recipe @red_add_i64 : i64 reduction_operator <add> init {
  // CHECK:         } combiner {
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
  // CHECK:       acc.reduction.recipe @red_max_f32 : f32 reduction_operator <max> init {
  // CHECK:         } combiner {
  // CHECK:         } destroy {

  // acc.terminator round-trips (pretty + generic) through mlir-opt's OpenACC
  // dialect. Upstream `acc.kernels` actually models its body via
  // `SingleBlockImplicitTerminator<"TerminatorOp">`, so mlir-opt re-emits the
  // body with the terminator on its own line.
  func.func @terminator_inside_kernels() {
    acc.kernels {
      acc.terminator
    }
    func.return
  }
  // CHECK:       func.func @terminator_inside_kernels() {
  // CHECK-NEXT:    acc.kernels {
  // CHECK-NEXT:      acc.terminator
  // CHECK-NEXT:    }

  // acc.enter_data — both pretty and generic-form spellings round-trip
  // through `mlir-opt`. The five-segment operand shape and the `async` /
  // `wait` UnitAttrs are the load-bearing properties. xDSL emits clauses
  // in upstream's td-definition order (if, async, wait_devnum, wait,
  // dataOperands); mlir-opt's `oilist` printer preserves input order.
  func.func @enter_data_minimal(%a : memref<10xf32>) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @enter_data_minimal(
  // CHECK:         acc.enter_data dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_async_bare(%a : memref<10xf32>) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data async dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @enter_data_async_bare(
  // CHECK:         acc.enter_data async dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_async_operand(%a : memref<10xf32>, %v : i64) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data async(%v : i64) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @enter_data_async_operand(
  // CHECK:         acc.enter_data async(%{{.*}} : i64) dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_wait_bare(%a : memref<10xf32>) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data wait dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @enter_data_wait_bare(
  // CHECK:         acc.enter_data wait dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_if(%a : memref<10xf32>, %c : i1) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data if(%c) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @enter_data_if(
  // CHECK:         acc.enter_data if(%{{.*}}) dataOperands(%{{.*}} : memref<10xf32>)

  func.func @enter_data_wait_devnum(%a : memref<10xf32>, %dn : i64, %w : i32) {
    %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.enter_data wait_devnum(%dn : i64) wait(%w : i32) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @enter_data_wait_devnum(
  // CHECK:         acc.enter_data wait_devnum(%{{.*}} : i64) wait(%{{.*}} : i32) dataOperands(%{{.*}} : memref<10xf32>)

  // acc.exit_data — twin of enter_data plus a `finalize` UnitAttr. Both
  // pretty and generic spellings round-trip through `mlir-opt`. Same
  // five-segment operand layout (if, async, wait_devnum, wait, dataOperands).
  func.func @exit_data_minimal(%a : memref<10xf32>) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @exit_data_minimal(
  // CHECK:         acc.exit_data dataOperands(%{{.*}} : memref<10xf32>)

  func.func @exit_data_async_finalize(%a : memref<10xf32>) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data async dataOperands(%d : memref<10xf32>) attributes {finalize}
    func.return
  }
  // CHECK:       func.func @exit_data_async_finalize(
  // CHECK:         acc.exit_data async dataOperands(%{{.*}} : memref<10xf32>) attributes {finalize}

  func.func @exit_data_async_operand(%a : memref<10xf32>, %v : i64) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data async(%v : i64) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @exit_data_async_operand(
  // CHECK:         acc.exit_data async(%{{.*}} : i64) dataOperands(%{{.*}} : memref<10xf32>)

  // Bare `wait` keyword (no operands) — UnitAttr code path through
  // `OperandsWithKeywordOnly`. Distinct from the operand-bearing
  // `wait_devnum`/`wait` form below.
  func.func @exit_data_wait_bare(%a : memref<10xf32>) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data wait dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @exit_data_wait_bare(
  // CHECK:         acc.exit_data wait dataOperands(%{{.*}} : memref<10xf32>)

  func.func @exit_data_if(%a : memref<10xf32>, %c : i1) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data if(%c) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @exit_data_if(
  // CHECK:         acc.exit_data if(%{{.*}}) dataOperands(%{{.*}} : memref<10xf32>)

  func.func @exit_data_wait_devnum(%a : memref<10xf32>, %dn : i64, %w : i32) {
    %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
    acc.exit_data wait_devnum(%dn : i64) wait(%w : i32) dataOperands(%d : memref<10xf32>)
    func.return
  }
  // CHECK:       func.func @exit_data_wait_devnum(
  // CHECK:         acc.exit_data wait_devnum(%{{.*}} : i64) wait(%{{.*}} : i32) dataOperands(%{{.*}} : memref<10xf32>)

  // acc.update — per-device-type async/wait shape mirroring `acc.parallel`,
  // plus an `ifPresent` UnitAttr. Defining ops accepted: `acc.update_device`,
  // `acc.update_host`, `acc.getdeviceptr`. Note that `mlir-opt`'s
  // `verifyDeviceTypeCountMatch` segfaults if `asyncOperandsDeviceType` is
  // not populated when there are async operands — xDSL's
  // `DeviceTypeOperandsWithKeywordOnly` directive auto-populates it.
  func.func @update_minimal(%a : memref<f32>) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update dataOperands(%d : memref<f32>)
    func.return
  }
  // CHECK:       func.func @update_minimal(
  // CHECK:         acc.update dataOperands(%{{.*}} : memref<f32>)

  // Bare `async` keyword on `acc.update` — distinct from `acc.exit_data async`:
  // it lands as `asyncOnly = [#acc.device_type<none>]` (an array attr) via
  // `DeviceTypeOperandsWithKeywordOnly`, *not* as a `UnitAttr`. This is the
  // op-specific keyword-only path on the per-device-type async/wait shape.
  func.func @update_async_bare(%a : memref<f32>) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update async dataOperands(%d : memref<f32>)
    func.return
  }
  // CHECK:       func.func @update_async_bare(
  // CHECK:         acc.update async dataOperands(%{{.*}} : memref<f32>)

  func.func @update_async_operand(%a : memref<f32>, %v : i64) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update async(%v : i64) dataOperands(%d : memref<f32>)
    func.return
  }
  // CHECK:       func.func @update_async_operand(
  // CHECK:         acc.update async(%{{.*}} : i64) dataOperands(%{{.*}} : memref<f32>)

  func.func @update_wait_devnum(%a : memref<f32>, %dn : i64, %w : i32) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update wait({devnum: %dn : i64, %w : i32}) dataOperands(%d : memref<f32>)
    func.return
  }
  // CHECK:       func.func @update_wait_devnum(
  // CHECK:         acc.update wait({devnum: %{{.*}} : i64, %{{.*}} : i32}) dataOperands(%{{.*}} : memref<f32>)

  func.func @update_if_present(%a : memref<f32>, %c : i1) {
    %d = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
    acc.update if(%c) dataOperands(%d : memref<f32>) attributes {ifPresent}
    func.return
  }
  // CHECK:       func.func @update_if_present(
  // CHECK:         acc.update if(%{{.*}}) dataOperands(%{{.*}} : memref<f32>) attributes {ifPresent}

  // acc.declare_enter / acc.declare_exit / acc.declare — declare family.
  // Defining ops accepted for `dataOperands`: copyin / copyout / create /
  // deviceptr / getdeviceptr / present / declare_device_resident /
  // declare_link (per upstream's `checkDeclareOperands` helper).
  func.func @declare_enter_basic(%a : memref<f32>) {
    %0 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32>
    %t = acc.declare_enter dataOperands(%0 : memref<f32>)
    acc.declare_exit token(%t) dataOperands(%0 : memref<f32>)
    func.return
  }
  // CHECK:       func.func @declare_enter_basic(
  // CHECK:         %[[CI:.*]] = acc.copyin varPtr(%{{.*}} : memref<f32>) -> memref<f32>
  // CHECK-NEXT:    %[[T:.*]] = acc.declare_enter dataOperands(%[[CI]] : memref<f32>)
  // CHECK-NEXT:    acc.declare_exit token(%[[T]]) dataOperands(%[[CI]] : memref<f32>)

  // acc.declare_exit's `token`-bearing form relaxes the at-least-one
  // operand requirement — proves the optional-operand AttrSizedOperandSegments
  // round-trips cleanly through mlir-opt's parser.
  func.func @declare_exit_token_only(%a : memref<f32>) {
    %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
    %t = acc.declare_enter dataOperands(%0 : memref<f32>)
    acc.declare_exit token(%t)
    func.return
  }
  // CHECK:       func.func @declare_exit_token_only(
  // CHECK:         acc.declare_exit token(%{{.*}})

  // acc.declare_exit without a `token`: only `dataOperands`. Mirrors
  // upstream's `getdeviceptr` example for device-resident tear-down.
  func.func @declare_exit_no_token(%a : memref<f32>) {
    %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
    acc.declare_exit dataOperands(%0 : memref<f32>)
    func.return
  }
  // CHECK:       func.func @declare_exit_no_token(
  // CHECK:         acc.declare_exit dataOperands(%{{.*}} : memref<f32>)

  // acc.declare — structured declare region. The body is the implicit
  // data region's lifetime (here empty; AnyRegion has no implicit
  // terminator).
  func.func @declare_region(%a : memref<f32>) {
    %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
    acc.declare dataOperands(%0 : memref<f32>) {
    }
    func.return
  }
  // CHECK:       func.func @declare_region(
  // CHECK:         acc.declare dataOperands(%{{.*}} : memref<f32>) {
  // CHECK-NEXT:    }

}
