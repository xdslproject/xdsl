// RUN: xdsl-opt -t csl %s | filecheck %s

"csl.module"() <{kind=#csl<module_kind program>}> ({
  // csl.func, csl.return
  csl.func @no_args_no_return() {
    csl.return
  }

  csl.func @no_args_return() -> f32 {
    %c = arith.constant 5.0 : f32
    csl.return %c : f32
  }

  csl.func @args_no_return(%a: i32, %b: i32) {
    csl.return
  }

  // csl.const_struct
  %empty_struct = "csl.const_struct"() : () -> !csl.comptime_struct
  %attribute_struct = "csl.const_struct"() <{items = {hello = 123 : f32}}> : () -> !csl.comptime_struct
  %const27 = arith.constant 27 : i16
  %ssa_struct = "csl.const_struct"(%const27) <{ssa_fields = ["val"]}> : (i16) -> !csl.comptime_struct

  %concat = "csl.concat_structs"(%empty_struct, %attribute_struct) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct

  %no_param_import = "csl.import_module"() <{module = "<mod>"}> : () -> !csl.imported_module
  %param_import = "csl.import_module"(%ssa_struct) <{module = "<mod>"}> : (!csl.comptime_struct) -> !csl.imported_module

  "csl.member_call"(%param_import) <{field = "foo"}> : (!csl.imported_module) -> ()
  "csl.member_call"(%param_import, %const27) <{field = "bar"}> : (!csl.imported_module, i16) -> ()
  %val2 = "csl.member_call"(%param_import) <{field = "baz"}> : (!csl.imported_module) -> (f32)

  %val3 = "csl.member_access"(%param_import) <{field = "f"}> :  (!csl.imported_module) -> (i32)


  csl.func @main() {
    "csl.call"() <{callee = @no_args_no_return}> : () -> ()
    "csl.call"(%val3, %val3) <{callee = @args_no_return}> : (i32, i32) -> ()
    %ret = "csl.call"() <{callee = @no_args_return}> : () -> (f32)

    csl.return
  }

  csl.func @name_collisions() {
    %align = arith.constant 0 : i32
    %and = arith.constant 0 : i32
    %bool = arith.constant 0 : i32
    %break = arith.constant 0 : i32
    %comptime_float = arith.constant 0 : i32
    %comptime_int = arith.constant 0 : i32
    %comptime_string = arith.constant 0 : i32
    %comptime_struct = arith.constant 0 : i32
    %const = arith.constant 0 : i32
    %continue = arith.constant 0 : i32
    %else = arith.constant 0 : i32
    %export = arith.constant 0 : i32
    %extern = arith.constant 0 : i32
    %f16 = arith.constant 0 : i32
    %f32 = arith.constant 0 : i32
    %false = arith.constant 0 : i32
    %fn = arith.constant 0 : i32
    %for = arith.constant 0 : i32
    %i16 = arith.constant 0 : i32
    %i32 = arith.constant 0 : i32
    %i64 = arith.constant 0 : i32
    %i8 = arith.constant 0 : i32
    %if = arith.constant 0 : i32
    %linkname = arith.constant 0 : i32
    %linksection = arith.constant 0 : i32
    %or = arith.constant 0 : i32
    %param = arith.constant 0 : i32
    %return = arith.constant 0 : i32
    %switch = arith.constant 0 : i32
    %task = arith.constant 0 : i32
    %true = arith.constant 0 : i32
    %u16 = arith.constant 0 : i32
    %u32 = arith.constant 0 : i32
    %u64 = arith.constant 0 : i32
    %u8 = arith.constant 0 : i32
    %var = arith.constant 0 : i32
    %void = arith.constant 0 : i32
    %while = arith.constant 0 : i32


    csl.return
  }

  csl.func @casts() {
    %constI32 = arith.constant 0 : i32
    %c16 = arith.constant 0 : i16
    %constU16 = csl.mlir.signedness_cast %c16 : i16 to ui16
    %constF32 = arith.constant 0 : f32
    %castIndex = "arith.index_cast"(%constU16) : (ui16) -> index
    %castF16 = "arith.sitofp"(%constI32) : (i32) -> f16
    %castI16 = "arith.fptosi"(%constF32) : (f32) -> i16
    %castF32 = "arith.extf"(%castF16) : (f16) -> f32
    %castF16again = "arith.truncf"(%constF32) : (f32) -> f16
    %castI16again = "arith.trunci"(%constI32) : (i32) -> i16
    %castI32again = "arith.extsi"(%castI16) : (i16) -> i32
    %castU32 = "arith.extui"(%constU16)  : (ui16) -> ui32
    csl.return
  }

  csl.func @comparisons() -> i1 {
    %int1 = arith.constant 0 : i32
    %int2 = arith.constant 1 : i32

    %float1 = arith.constant 0.0 : f32
    %float2 = arith.constant 1.5 : f32

    %intEq  = arith.cmpi eq,  %int1, %int2 : i32
    %intNe  = arith.cmpi ne,  %int1, %int2 : i32
    %intSlt = arith.cmpi slt, %int1, %int2 : i32
    %intSle = arith.cmpi sle, %int1, %int2 : i32
    %intSgt = arith.cmpi sgt, %int1, %int2 : i32
    %intSge = arith.cmpi sge, %int1, %int2 : i32
    %intUlt = arith.cmpi ult, %int1, %int2 : i32
    %intUle = arith.cmpi ule, %int1, %int2 : i32
    %intUgt = arith.cmpi ugt, %int1, %int2 : i32
    %intUge = arith.cmpi uge, %int1, %int2 : i32

    %floatFalse = arith.cmpf false, %float1, %float2 : f32
    %floatOeq   = arith.cmpf oeq,   %float1, %float2 : f32
    %floatOgt   = arith.cmpf ogt,   %float1, %float2 : f32
    %floatOge   = arith.cmpf oge,   %float1, %float2 : f32
    %floatOlt   = arith.cmpf olt,   %float1, %float2 : f32
    %floatOle   = arith.cmpf ole,   %float1, %float2 : f32
    %floatOne   = arith.cmpf one,   %float1, %float2 : f32
    %floatUeq   = arith.cmpf ueq,   %float1, %float2 : f32
    %floatUgt   = arith.cmpf ugt,   %float1, %float2 : f32
    %floatUge   = arith.cmpf uge,   %float1, %float2 : f32
    %floatUlt   = arith.cmpf ult,   %float1, %float2 : f32
    %floatUle   = arith.cmpf ule,   %float1, %float2 : f32
    %floatUne   = arith.cmpf une,   %float1, %float2 : f32
    %floatTrue  = arith.cmpf true,  %float1, %float2 : f32

    %or_ = arith.ori %intSle, %floatUgt : i1
    %and_ = arith.andi %or_, %floatOge : i1

    csl.return %and_ : i1
  }

  csl.func @select() -> !csl<dsd mem1d_dsd> {
    %value1 = arith.constant 100 : i32
    %value2 = arith.constant 200 : i32
    %toggle = arith.constant 1 : i1
    %A = memref.get_global @A : memref<24xf32>
    %dsd1 = "csl.get_mem_dsd"(%A, %value1) : (memref<24xf32>, i32) -> !csl<dsd mem1d_dsd>
    %dsd2 = "csl.get_mem_dsd"(%A, %value2) : (memref<24xf32>, i32) -> !csl<dsd mem1d_dsd>
    %selected_dsd = arith.select %toggle, %dsd1, %dsd2 : !csl<dsd mem1d_dsd>
    csl.return %selected_dsd : !csl<dsd mem1d_dsd>
  }

  csl.func @constants() {
    %inline_const = arith.constant 100 : i32

    %1 = "csl.constants"(%const27, %const27) : (i16, i16) -> memref<?xi16>

    %2 = "csl.constants"(%const27, %const27) <{is_const}> : (i16, i16) -> memref<?xi16>

    %3 = "csl.constants"(%const27, %inline_const) <{is_const}> : (i16, i32) -> memref<?xi32>

    %4 = "csl.constants"(%inline_const, %inline_const) <{is_const}> : (i32, i32) -> memref<?xi32>

    %5 = "csl.zeros"(%const27) <{is_const}> : (i16) -> memref<?xi16>

    csl.return
  }


  csl.task @data_task(%arg: f32) attributes {kind = #csl<task_kind data>, id = 0 : ui5} {
    csl.return
  }

  csl.task @local_task() attributes {kind = #csl<task_kind local>, id = 1 : ui5} {
    csl.return
  }

  csl.task @control_task() attributes {kind = #csl<task_kind control>, id = 42 : ui6} {
    csl.return
  }

  csl.task @data_task_no_bind(%arg: f32) attributes {kind = #csl<task_kind data>} {
    csl.return
  }

  csl.task @local_task_no_bind() attributes {kind = #csl<task_kind local>} {
    csl.return
  }

  csl.task @control_task_no_bind() attributes {kind = #csl<task_kind control>} {
    csl.return
  }

  csl.func @variables() {
    %one = arith.constant 1 : i32
    %variable_with_default = "csl.variable"() <{default = 42 : i32}> : () -> !csl.var<i32>
    %variable = "csl.variable"() : () -> !csl.var<i32>
    %value = "csl.load_var"(%variable_with_default) : (!csl.var<i32>) -> i32
    %new_value = arith.addi %value, %one : i32
    "csl.store_var"(%variable_with_default, %new_value) : (!csl.var<i32>, i32) -> ()
    "csl.store_var"(%variable, %new_value) : (!csl.var<i32>, i32) -> ()

    csl.return
  }

  "memref.global"() {"sym_name" = "uninit_array", "type" = memref<10xf32>, "sym_visibility" = "public", "initial_value"} : () -> ()
  "memref.global"() {"sym_name" = "global_array", "type" = memref<10xf32>, "sym_visibility" = "public", "initial_value" = dense<4.5> : tensor<1xf32>} : () -> ()
  "memref.global"() {"sym_name" = "const_array", "type" = memref<10xi32>, "sym_visibility" = "public", "constant", "initial_value" = dense<10> : tensor<1xi32>} : () -> ()


  %uninit_array = memref.get_global @uninit_array : memref<10xf32>
  %global_array = memref.get_global @global_array : memref<10xf32>
  %const_array = memref.get_global @const_array : memref<10xi32>

  %literal_array = arith.constant dense<[1.500000e+00, 2.500000e+00, 3.500000e+00]> : memref<3xf32>
  %literal_array_w_zeros = arith.constant dense<[1.500000e+00, 0, 3.500000e+00, 0]> : memref<4xf32>

  %uninit_ptr = "csl.addressof"(%uninit_array) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %global_ptr = "csl.addressof"(%global_array) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %const_ptr  = "csl.addressof"(%const_array) : (memref<10xi32>) -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>

  %ptr_to_arr = "csl.addressof"(%uninit_array) : (memref<10xf32>) -> !csl.ptr<memref<10xf32>, #csl<ptr_kind single>, #csl<ptr_const var>>
  %ptr_to_val = "csl.addressof"(%const27) : (i16) -> !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const const>>

  %ptr_1_fn = "csl.addressof_fn"() <{fn_name = @args_no_return}> : () -> !csl.ptr<(i32, i32) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
  %ptr_2_fn = "csl.addressof_fn"() <{fn_name = @no_args_return}> : () -> !csl.ptr<() -> (f32), #csl<ptr_kind single>, #csl<ptr_const const>>
  %dir_test = "csl.get_dir"() <{"dir" = #csl<dir_kind north>}> : () -> !csl.direction
  // putting dir_test in a struct to make sure it gets correctly inlined
  %struct_with_dir = "csl.const_struct"(%dir_test) <{ssa_fields = ["dir"]}> : (!csl.direction) -> !csl.comptime_struct



  "csl.export"(%global_ptr) <{
    type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>,
    var_name = "ptr_name"
  }> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()

  "csl.export"(%const_ptr) <{
    type = !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>,
    var_name = "another_ptr"
  }> : (!csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> ()

  "csl.export"() <{type = () -> (), var_name = @no_args_no_return}> : () -> ()

  "csl.export"() <{type = (i32, i32) -> (), var_name = @args_no_return}> : () -> ()

    %cst15 = arith.constant 15 : i32
    %col  = "csl.get_color"(%cst15) : (i32) -> !csl.color

    "csl.rpc"(%col) : (!csl.color) -> ()



"memref.global"() {"sym_name" = "A", "type" = memref<24xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "x", "type" = memref<6xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "b", "type" = memref<4xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "y", "type" = memref<4xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()

%thing = "csl.import_module"() <{module = "<thing>"}> : () -> !csl.imported_module


csl.func @initialize() {
  %lb = arith.constant   0 : i16
  %ub = arith.constant  24 : i16
  %step = arith.constant 1 : i16

  // call without result
  "csl.member_call"(%thing, %lb, %ub) <{field = "some_func"}> : (!csl.imported_module, i16, i16) -> ()

  // call with result
  %res = "csl.member_call"(%thing, %lb, %ub) <{field = "some_func"}> : (!csl.imported_module, i16, i16) -> (i32)

  // member access
  %11 = "csl.member_access"(%thing) <{field = "some_field"}> : (!csl.imported_module) -> !csl.comptime_struct

  %0 = arith.constant 3.5 : f32
  %v0 = arith.constant 2.5 : f16

  %c44 = arith.constant 44 : i32
  %u32cst = csl.mlir.signedness_cast %c44 : i32 to ui32

  %A = memref.get_global @A : memref<24xf32>
  %x = memref.get_global @x : memref<6xf32>
  %b = memref.get_global @b : memref<4xf32>
  %y = memref.get_global @y : memref<4xf32>

  scf.for %idx = %lb to %ub step %step : i16 {
    %idx_f32 = arith.sitofp %idx : i16 to f32
    %idx_index = "arith.index_cast"(%idx) : (i16) -> index
    memref.store %idx_f32, %A[%idx_index] : memref<24xf32>
  }

  %ub_6 = arith.constant 6 : i16

  scf.for %j = %lb to %ub_6 step %step : i16 {
    %val = arith.constant 1.0 : f32
    %j_idx = "arith.index_cast"(%j) : (i16) -> index
    memref.store %val, %x[%j_idx] : memref<6xf32>
  }

  %ub_4 = arith.constant 6 : i16

  scf.for %i = %lb to %ub_4 step %step : i16 {
    %c2 = arith.constant 2.0 : f32
    %c0 = arith.constant 0.0 : f32
    %i_idx = "arith.index_cast"(%i) : (i16) -> index
    memref.store %c2, %b[%i_idx] : memref<4xf32>
    memref.store %c0, %y[%i_idx] : memref<4xf32>
  }

  csl.return
}
csl.func @gemv() {
  %A = memref.get_global @A : memref<24xf32>
  %x = memref.get_global @x : memref<6xf32>
  %b = memref.get_global @b : memref<4xf32>
  %y = memref.get_global @y : memref<4xf32>

  %lb    = arith.constant 0 : index
  %step  = arith.constant 1 : index
  %ub_6  = arith.constant 6 : index
  %ub_4  = arith.constant 4 : index
  scf.for %i = %lb to %ub_4 step %step {

    %tmp_0 = arith.constant 0.0 : f32

    %tmp = scf.for %j = %lb to %ub_6 step %step
          iter_args(%tmp_iter = %tmp_0) -> (f32) {

      %ix6 = arith.muli %i, %ub_6 : index
      %ix6pj = arith.addi %ix6, %j : index
      %A_loaded = memref.load %A[%ix6pj] : memref<24xf32>
      %x_loaded = memref.load %x[%j] : memref<6xf32>

      %Axx = arith.mulf %A_loaded, %x_loaded : f32
      %tmp_next = arith.addf %tmp_iter, %Axx : f32
      scf.yield %tmp_next : f32

    }
    %bi = memref.load %b[%i] : memref<4xf32>
    %tmp_plus_bi = arith.addf %tmp, %bi : f32
    memref.store %tmp_plus_bi, %y[%i] : memref<4xf32>
  }

  csl.return
}

csl.func @ctrlflow() {
  %0 = arith.constant 0 : i1
  %1 = arith.constant 1 : i1
  %i32_value = arith.constant 100 : i32
  "scf.if"(%0) ({
    %2 = arith.constant 2 : i32
    scf.yield
  }, {
    %3 = arith.constant 3 : i32
    scf.yield
  }) : (i1) -> ()

  "scf.if"(%1) ({
    %4 = arith.constant 4 : i32
    scf.yield
  }, {
    scf.yield
  }) : (i1) -> ()

  %i32ret = "scf.if"(%0) ({
    %5 = arith.constant 111 : i32
    scf.yield %5 : i32
  }, {
    %6 = arith.constant 222 : i32
    scf.yield %6 : i32
  }) : (i1) -> (i32)


  csl.return
}

%tsc = "csl.import_module"() <{"module" = "<time>"}> : () -> !csl.imported_module
%tsc_buf = "csl.zeros"() : () -> memref<6xui16>

csl.func @timestamp_start() {
  %addr_of_1 = "csl.addressof"(%tsc_buf) : (memref<6xui16>) -> !csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>
  %ptrcast_1 = "csl.ptrcast"(%addr_of_1) : (!csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>) -> !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>
  "csl.member_call"(%tsc) <{"field" = "enable_tsc"}> : (!csl.imported_module) -> ()
  "csl.member_call"(%tsc, %ptrcast_1) <{"field" = "get_timestamp"}> : (!csl.imported_module, !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
  csl.return
}

csl.func @timestamp_stop() {
  %three = arith.constant 3 : index
  %ld = memref.load %tsc_buf[%three] : memref<6xui16>
  %addr_of_2 = "csl.addressof"(%ld) : (ui16) -> !csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>
  %ptrcast_2 = "csl.ptrcast"(%addr_of_2) : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>) -> !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>
  "csl.member_call"(%tsc, %ptrcast_2) <{"field" = "get_timestamp"}> : (!csl.imported_module, !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
  csl.return
}

csl.func @builtins() {
  %c10a = arith.constant 10 : i8
  %i8_value = csl.mlir.signedness_cast %c10a : i8 to si8
  %c10b = arith.constant 10 : i16
  %i16_value = csl.mlir.signedness_cast %c10b : i16 to si16
  %c12 = arith.constant 12 : i16
  %u16_value = csl.mlir.signedness_cast %c12 : i16 to ui16
  %c100 = arith.constant 100 : i32
  %i32_value = csl.mlir.signedness_cast %c100 : i32 to si32
  %c120 = arith.constant 120 : i32
  %u32_value = csl.mlir.signedness_cast %c120 : i32 to ui32
  %f16_value = arith.constant 7.0 : f16
  %f32_value = arith.constant 8.0 : f32
  %three = arith.constant 3 : i16
  %col_1 = "csl.get_color"(%three) : (i16) -> !csl.color
  %f16_pointer = "csl.addressof"(%f16_value) : (f16) -> !csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>
  %f32_pointer = "csl.addressof"(%f32_value) : (f32) -> !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>
  %i16_pointer = "csl.addressof"(%i16_value) : (si16) -> !csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>
  %i32_pointer = "csl.addressof"(%i32_value) : (si32) -> !csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>
  %u16_pointer = "csl.addressof"(%u16_value) : (ui16) -> !csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>
  %u32_pointer = "csl.addressof"(%u32_value) : (ui32) -> !csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>

  %A = memref.get_global @A : memref<24xf32>
  %dsd_2d = "csl.get_mem_dsd"(%A, %i32_value, %i32_value) <{"tensor_access" = affine_map<(d0, d1) -> (((d0 * 3) + 1), ((d1 * 4) + 2))>}> : (memref<24xf32>, si32, si32) -> !csl<dsd mem4d_dsd>
  %dest_dsd = "csl.get_mem_dsd"(%A, %i32_value) : (memref<24xf32>, si32) -> !csl<dsd mem1d_dsd>
  %src_dsd1 = "csl.get_mem_dsd"(%A, %i32_value) : (memref<24xf32>, si32) -> !csl<dsd mem1d_dsd>
  %src_dsd2 = "csl.get_mem_dsd"(%A, %i32_value) : (memref<24xf32>, si32) -> !csl<dsd mem1d_dsd>

  %dsd_1d2 = "csl.set_dsd_base_addr"(%dest_dsd, %A) : (!csl<dsd mem1d_dsd>, memref<24xf32>) -> !csl<dsd mem1d_dsd>
  %dsd_1d3 = "csl.increment_dsd_offset"(%dsd_1d2, %c10b) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
  %dsd_1d4 = "csl.set_dsd_length"(%dsd_1d3, %c12) : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
  %dsd_1d5 = "csl.set_dsd_stride"(%dsd_1d4, %c10a) : (!csl<dsd mem1d_dsd>, i8) -> !csl<dsd mem1d_dsd>

  %fabin_dsd = "csl.get_fab_dsd"(%i32_value) <{"fabric_color" = 2 : ui5 , "queue_id" = 0 : i3}> : (si32) -> !csl<dsd fabin_dsd>
  %fabout_dsd = "csl.get_fab_dsd"(%i32_value) <{"fabric_color" = 3 : ui5 , "queue_id" = 1 : i3, "control"= true, "wavelet_index_offset" = false}>: (si32) -> !csl<dsd fabout_dsd>

  %zero_stride_dsd = "csl.get_mem_dsd"(%A, %i16_value, %i16_value, %i16_value) <{"tensor_access" = affine_map<(d0, d1, d2) -> (d2)>}> : (memref<24xf32>, si16, si16, si16) -> !csl<dsd mem4d_dsd>
  %B = memref.get_global @B : memref<3x64xf32>
  %oned_access_into_twod = "csl.get_mem_dsd"(%B, %i16_value) <{"tensor_access" = affine_map<(d0) -> (1, d0)>}> : (memref<3x64xf32>, si16) -> !csl<dsd mem1d_dsd>

  "csl.add16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.addc16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
  "csl.and16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
  "csl.clz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
  "csl.ctz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
  "csl.fabsh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
  "csl.fabss"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
  "csl.faddh"(%f16_pointer, %f16_value, %src_dsd1)  : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
  "csl.faddhs"(%f32_pointer, %f32_value, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
  "csl.fadds"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.fh2s"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
  "csl.fh2xp16"(%i16_pointer, %f16_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
  "csl.fmacs" (%dest_dsd, %src_dsd1, %src_dsd2, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
  "csl.fmaxh"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.fmaxs"(%dest_dsd,    %f32_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
  "csl.fmovh"(%f16_pointer, %src_dsd1)  : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
  "csl.fmovs"(%dest_dsd,    %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
  "csl.fmulh"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.fmuls"(%f32_pointer, %f32_value, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
  "csl.fnegh"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.fnegs"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
  "csl.fnormh"(%f16_pointer, %f16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
  "csl.fnorms"(%f32_pointer, %f32_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()
  "csl.fs2h"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.fs2xp16"(%i16_pointer, %f32_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()
  "csl.fscaleh"(%f16_pointer, %f16_value, %i16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, si16) -> ()
  "csl.fscales"(%f32_pointer, %f32_value, %i16_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, si16) -> ()
  "csl.fsubh"(%f16_pointer, %f16_value, %src_dsd1)  : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
  "csl.fsubs"(%f32_pointer, %f32_value, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
  "csl.mov16"(%u16_pointer, %src_dsd1)  : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
  "csl.mov32"(%i32_pointer, %src_dsd1)  : (!csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
  "csl.or16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
  "csl.popcnt"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.sar16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.sll16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
  "csl.slr16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
  "csl.sub16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
  "csl.xor16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.xp162fh"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
  "csl.xp162fs"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()

  csl.activate data, 0 : ui6
  csl.activate local, 1 : ui6
  csl.activate control, 42 : ui6

  csl.return
}

}) {sym_name = "program.csl"} : () -> ()


"csl.module"() <{kind=#csl<module_kind layout>}> ({
  %x_dim = "csl.param"() <{param_name = "param_1"}> : () -> i32
  %init = arith.constant 3.5 : f16
  %p2 = "csl.param"(%init) <{param_name = "param_2"}> : (f16) -> f16

  csl.layout {
    %y_dim = arith.constant 6 : i32
    "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()

    %x_coord0 = arith.constant 0 : i32
    %y_coord = arith.constant 0 : i32
    "csl.set_tile_code"(%x_coord0, %y_coord) <{file = "file.csl"}> : (i32, i32) -> ()

    %params = "csl.const_struct"(){items = {hello = 123 : i32}} : () -> !csl.comptime_struct
    %x_coord1 = arith.constant 1 : i32
    "csl.set_tile_code"(%x_coord1, %y_coord, %params) <{file = "program.csl"}> : (i32, i32, !csl.comptime_struct) -> ()

  }
}) {sym_name = "layout.csl"} : () -> ()

// CHECK: // FILE: program.csl
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn no_args_no_return() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn no_args_return() f32 {
// CHECK-NEXT:   return 5.0;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn args_no_return(a : i32, b : i32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: const empty_struct : comptime_struct = .{
// CHECK-NEXT: };
// CHECK-NEXT: const attribute_struct : comptime_struct = .{
// CHECK-NEXT:   .hello = 123.0,
// CHECK-NEXT: };
// CHECK-NEXT: const const27 : i16 = 27;
// CHECK-NEXT: const ssa_struct : comptime_struct = .{
// CHECK-NEXT:   .val = const27,
// CHECK-NEXT: };
// CHECK-NEXT: const concat : comptime_struct = @concat_structs(empty_struct, attribute_struct);
// CHECK-NEXT: const no_param_import : imported_module = @import_module("<mod>");
// CHECK-NEXT: const param_import : imported_module = @import_module("<mod>", ssa_struct);
// CHECK-NEXT: param_import.foo();
// CHECK-NEXT: param_import.bar(const27);
// CHECK-NEXT: const val2 : f32 = param_import.baz();
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn main() void {
// CHECK-NEXT:   no_args_no_return();
// CHECK-NEXT:   args_no_return(param_import.f, param_import.f);
// CHECK-NEXT:   const ret : f32 = no_args_return();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn name_collisions() void {
// CHECK-NEXT:   const align0 : i32 = 0;
// CHECK-NEXT:   const and1 : i32 = 0;
// CHECK-NEXT:   const bool2 : i32 = 0;
// CHECK-NEXT:   const break3 : i32 = 0;
// CHECK-NEXT:   const comptime_float4 : i32 = 0;
// CHECK-NEXT:   const comptime_int5 : i32 = 0;
// CHECK-NEXT:   const comptime_string6 : i32 = 0;
// CHECK-NEXT:   const comptime_struct7 : i32 = 0;
// CHECK-NEXT:   const const8 : i32 = 0;
// CHECK-NEXT:   const continue9 : i32 = 0;
// CHECK-NEXT:   const else10 : i32 = 0;
// CHECK-NEXT:   const export11 : i32 = 0;
// CHECK-NEXT:   const extern12 : i32 = 0;
// CHECK-NEXT:   const f1613 : i32 = 0;
// CHECK-NEXT:   const f3214 : i32 = 0;
// CHECK-NEXT:   const false15 : i32 = 0;
// CHECK-NEXT:   const fn16 : i32 = 0;
// CHECK-NEXT:   const for17 : i32 = 0;
// CHECK-NEXT:   const i1618 : i32 = 0;
// CHECK-NEXT:   const i3219 : i32 = 0;
// CHECK-NEXT:   const i6420 : i32 = 0;
// CHECK-NEXT:   const i821 : i32 = 0;
// CHECK-NEXT:   const if22 : i32 = 0;
// CHECK-NEXT:   const linkname23 : i32 = 0;
// CHECK-NEXT:   const linksection24 : i32 = 0;
// CHECK-NEXT:   const or25 : i32 = 0;
// CHECK-NEXT:   const param26 : i32 = 0;
// CHECK-NEXT:   const return27 : i32 = 0;
// CHECK-NEXT:   const switch28 : i32 = 0;
// CHECK-NEXT:   const task29 : i32 = 0;
// CHECK-NEXT:   const true30 : i32 = 0;
// CHECK-NEXT:   const u1631 : i32 = 0;
// CHECK-NEXT:   const u3232 : i32 = 0;
// CHECK-NEXT:   const u6433 : i32 = 0;
// CHECK-NEXT:   const u834 : i32 = 0;
// CHECK-NEXT:   const var35 : i32 = 0;
// CHECK-NEXT:   const void36 : i32 = 0;
// CHECK-NEXT:   const while37 : i32 = 0;
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn casts() void {
// CHECK-NEXT:   const castIndex : i32 = @as(i32, @as(u16, 0));
// CHECK-NEXT:   const castF32 : f32 = @as(f32, @as(f16, 0));
// CHECK-NEXT:   const castF16again : f16 = @as(f16, 0.0);
// CHECK-NEXT:   const castI16again : i16 = @as(i16, 0);
// CHECK-NEXT:   const castI32again : i32 = @as(i32, @as(i16, 0.0));
// CHECK-NEXT:   const castU32 : u32 = @as(u32, @as(u16, 0));
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn comparisons() bool {
// CHECK-NEXT:   const intEq : bool = 0  ==  1;
// CHECK-NEXT:   const intNe : bool = 0  !=  1;
// CHECK-NEXT:   const intSlt : bool = 0  <  1;
// CHECK-NEXT:   const intSgt : bool = 0  >  1;
// CHECK-NEXT:   const intSge : bool = 0  >=  1;
// CHECK-NEXT:   const intUlt : bool = 0  <  1;
// CHECK-NEXT:   const intUle : bool = 0  <=  1;
// CHECK-NEXT:   const intUgt : bool = 0  >  1;
// CHECK-NEXT:   const intUge : bool = 0  >=  1;
// CHECK-NEXT:   const floatFalse : bool = false;
// CHECK-NEXT:   const floatOeq : bool = 0.0  ==  1.5;
// CHECK-NEXT:   const floatOgt : bool = 0.0  >  1.5;
// CHECK-NEXT:   const floatOlt : bool = 0.0  <  1.5;
// CHECK-NEXT:   const floatOle : bool = 0.0  <=  1.5;
// CHECK-NEXT:   const floatOne : bool = 0.0  !=  1.5;
// CHECK-NEXT:   const floatUeq : bool = 0.0  ==  1.5;
// CHECK-NEXT:   const floatUge : bool = 0.0  >=  1.5;
// CHECK-NEXT:   const floatUlt : bool = 0.0  <  1.5;
// CHECK-NEXT:   const floatUle : bool = 0.0  <=  1.5;
// CHECK-NEXT:   const floatUne : bool = 0.0  !=  1.5;
// CHECK-NEXT:   const floatTrue : bool = true;
// CHECK-NEXT:   return (((0  <=  1) or (0.0  >  1.5)) and (0.0  >=  1.5));
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn select() mem1d_dsd {
// CHECK-NEXT:   const dsd1 : mem1d_dsd = @get_dsd( mem1d_dsd, .{
// CHECK-NEXT:     .tensor_access = | d0 | { 100 } -> A[ d0 ]
// CHECK-NEXT:   });
// CHECK-NEXT:   const dsd2 : mem1d_dsd = @get_dsd( mem1d_dsd, .{
// CHECK-NEXT:     .tensor_access = | d0 | { 200 } -> A[ d0 ]
// CHECK-NEXT:   });
// CHECK-NEXT:   return (if (true) dsd1 else dsd2);
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn constants() void {
// CHECK-NEXT:   var v0 : [const27]i16 = @constants([const27]i16, const27);
// CHECK-NEXT:   const v1 : [const27]i16 = @constants([const27]i16, const27);
// CHECK-NEXT:   const v2 : [const27]i32 = @constants([const27]i32, 100);
// CHECK-NEXT:   const v3 : [100]i32 = @constants([100]i32, 100);
// CHECK-NEXT:   const v4 : [const27]i16 = @zeros([const27]i16);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: task data_task(arg : f32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_data_task(data_task, @get_data_task_id(@get_color(0)));
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: task local_task() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_local_task(local_task, @get_local_task_id(1));
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: task control_task() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_control_task(control_task, @get_control_task_id(42));
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: task data_task_no_bind(arg0 : f32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: task local_task_no_bind() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: task control_task_no_bind() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn variables() void {
// CHECK-NEXT:   var variable_with_default : i32 = 42;
// CHECK-NEXT:   var variable : i32;
// CHECK-NEXT:   const value : i32 = variable_with_default;
// CHECK-NEXT:   variable_with_default = (value + 1);
// CHECK-NEXT:   variable = (value + 1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: var uninit_array : [10]f32;
// CHECK-NEXT: var global_array : [10]f32 = @constants([10]f32, 4.5);
// CHECK-NEXT: const const_array : [10]i32 = @constants([10]i32, 10);
// CHECK-NEXT: const literal_array : [3]f32 = [3]f32 { 1.5, 2.5, 3.5 };
// CHECK-NEXT: const literal_array_w_zeros : [4]f32 = [4]f32 { 1.5, 0.0, 3.5, 0.0 };
// CHECK-NEXT: var uninit_ptr : [*]f32 = &uninit_array;
// CHECK-NEXT: var global_ptr : [*]f32 = &global_array;
// CHECK-NEXT: const const_ptr : [*]const i32 = &const_array;
// CHECK-NEXT: var ptr_to_arr : *[10]f32 = &uninit_array;
// CHECK-NEXT: const ptr_to_val : *const i16 = &const27;
// CHECK-NEXT: const ptr_1_fn : *const fn(i32, i32) void = &args_no_return;
// CHECK-NEXT: const ptr_2_fn : *const fn() f32 = &no_args_return;
// CHECK-NEXT: const struct_with_dir : comptime_struct = .{
// CHECK-NEXT:   .dir = NORTH,
// CHECK-NEXT: };
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(global_ptr, "ptr_name");
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(const_ptr, "another_ptr");
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(no_args_no_return, "no_args_no_return");
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(args_no_return, "args_no_return");
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @rpc(@get_data_task_id(@get_color(15)));
// CHECK-NEXT: }
// CHECK-NEXT: var A : [24]f32 = @constants([24]f32, 0);
// CHECK-NEXT: var x : [6]f32 = @constants([6]f32, 0);
// CHECK-NEXT: var b : [4]f32 = @constants([4]f32, 0);
// CHECK-NEXT: var y : [4]f32 = @constants([4]f32, 0);
// CHECK-NEXT: const thing : imported_module = @import_module("<thing>");
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn initialize() void {
// CHECK-NEXT:   thing.some_func(0, 24);
// CHECK-NEXT:   const res : i32 = thing.some_func(0, 24);
// CHECK-NEXT:   const v1 : comptime_struct = thing.some_field;
// CHECK-NEXT:   const v2 : f32 = 3.5;
// CHECK-NEXT:   const v0 : f16 = 2.5;
// CHECK-NEXT:   const u32cst : u32 = @as(u32, 44);
// CHECK-NEXT: {{ *}}
// CHECK-NEXT:   for(@range(i16, 0, 24, 1)) |idx| {
// CHECK-NEXT:     A[@as(i32, idx)] = @as(f32, idx);
// CHECK-NEXT:   }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT:   for(@range(i16, 0, 6, 1)) |j| {
// CHECK-NEXT:     x[@as(i32, j)] = 1.0;
// CHECK-NEXT:   }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT:   for(@range(i16, 0, 6, 1)) |i| {
// CHECK-NEXT:     b[@as(i32, i)] = 2.0;
// CHECK-NEXT:     y[@as(i32, i)] = 0.0;
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn gemv() void {
// CHECK-NEXT: {{ *}}
// CHECK-NEXT:   for(@range(i32, 0, 4, 1)) |i| {
// CHECK-NEXT:     var tmp : f32 = 0.0;
// CHECK-NEXT: {{ *}}
// CHECK-NEXT:     for(@range(i32, 0, 6, 1)) |j| {
// CHECK-NEXT:       tmp = tmp + ((A[((i * 6) + j)]) * (x[j]));
// CHECK-NEXT:     }
// CHECK-NEXT:     y[i] = (tmp + (b[i]));
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn ctrlflow() void {
// CHECK-NEXT:   const i32_value : i32 = 100;
// CHECK-NEXT:   if (false) {
// CHECK-NEXT:     const v1 : i32 = 2;
// CHECK-NEXT:   }
// CHECK-NEXT:   else {
// CHECK-NEXT:     const v1 : i32 = 3;
// CHECK-NEXT:   }
// CHECK-NEXT:   if (true) {
// CHECK-NEXT:     const v1 : i32 = 4;
// CHECK-NEXT:   }
// CHECK-NEXT:   var i32ret : i32;
// CHECK-NEXT:   if (false) {
// CHECK-NEXT:     i32ret = 111;
// CHECK-NEXT:   }
// CHECK-NEXT:   else {
// CHECK-NEXT:     i32ret = 222;
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: const tsc : imported_module = @import_module("<time>");
// CHECK-NEXT: var tsc_buf : [6]u16 = @zeros([6]u16);
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn timestamp_start() void {
// CHECK-NEXT:   var addr_of : [*]u16 = &tsc_buf;
// CHECK-NEXT:   tsc.enable_tsc();
// CHECK-NEXT:   tsc.get_timestamp(@ptrcast(*[3]u16, addr_of));
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn timestamp_stop() void {
// CHECK-NEXT:   var addr_of : *u16 = &(tsc_buf[3]);
// CHECK-NEXT:   tsc.get_timestamp(@ptrcast(*[3]u16, addr_of));
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: {{ *}}
// CHECK-NEXT: fn builtins() void {
// CHECK-NEXT:   const i8_value : i8 = @as(i8, 10);
// CHECK-NEXT:   const i16_value : i16 = @as(i16, 10);
// CHECK-NEXT:   const u16_value : u16 = @as(u16, 12);
// CHECK-NEXT:   const i32_value : i32 = @as(i32, 100);
// CHECK-NEXT:   const u32_value : u32 = @as(u32, 120);
// CHECK-NEXT:   const f16_value : f16 = 7.0;
// CHECK-NEXT:   const f32_value : f32 = 8.0;
// CHECK-NEXT:   const col : color = @get_color(3);
// CHECK-NEXT:   var f16_pointer : *f16 = &f16_value;
// CHECK-NEXT:   var f32_pointer : *f32 = &f32_value;
// CHECK-NEXT:   var i16_pointer : *i16 = &i16_value;
// CHECK-NEXT:   var i32_pointer : *i32 = &i32_value;
// CHECK-NEXT:   var u16_pointer : *u16 = &u16_value;
// CHECK-NEXT:   var u32_pointer : *u32 = &u32_value;
// CHECK-NEXT:   const dsd_2d : mem4d_dsd = @get_dsd( mem4d_dsd, .{
// CHECK-NEXT:     .tensor_access = | d0, d1 | { i32_value, i32_value } -> A[ ((d0 * 3) + 1), ((d1 * 4) + 2) ]
// CHECK-NEXT:   });
// CHECK-NEXT:   const dest_dsd : mem1d_dsd = @get_dsd( mem1d_dsd, .{
// CHECK-NEXT:     .tensor_access = | d0 | { i32_value } -> A[ d0 ]
// CHECK-NEXT:   });
// CHECK-NEXT:   const src_dsd1 : mem1d_dsd = @get_dsd( mem1d_dsd, .{
// CHECK-NEXT:     .tensor_access = | d0 | { i32_value } -> A[ d0 ]
// CHECK-NEXT:   });
// CHECK-NEXT:   const src_dsd2 : mem1d_dsd = @get_dsd( mem1d_dsd, .{
// CHECK-NEXT:     .tensor_access = | d0 | { i32_value } -> A[ d0 ]
// CHECK-NEXT:   });
// CHECK-NEXT:   const dsd_1d2 : mem1d_dsd = @set_dsd_base_addr(dest_dsd, A);
// CHECK-NEXT:   const dsd_1d3 : mem1d_dsd = @increment_dsd_offset(dsd_1d2, 10, f32);
// CHECK-NEXT:   const dsd_1d4 : mem1d_dsd = @set_dsd_length(dsd_1d3, 12);
// CHECK-NEXT:   const dsd_1d5 : mem1d_dsd = @set_dsd_stride(dsd_1d4, 10);
// CHECK-NEXT:   const fabin_dsd : fabin_dsd = @get_dsd(fabin_dsd, .{
// CHECK-NEXT:     .extent = i32_value,
// CHECK-NEXT:     .input_queue = @get_input_queue(0),
// CHECK-NEXT:     .fabric_color = 2 : ui5,
// CHECK-NEXT:   }});
// CHECK-NEXT:   const fabout_dsd : fabout_dsd = @get_dsd(fabout_dsd, .{
// CHECK-NEXT:     .extent = i32_value,
// CHECK-NEXT:     .output_queue = @get_output_queue(1),
// CHECK-NEXT:     .fabric_color = 3 : ui5,
// CHECK-NEXT:     .wavelet_index_offset = false,
// CHECK-NEXT:     .control = true,
// CHECK-NEXT:   }});
// CHECK-NEXT:   const zero_stride_dsd : mem4d_dsd = @get_dsd( mem4d_dsd, .{
// CHECK-NEXT:     .tensor_access = | d0, d1, d2 | { i16_value, i16_value, i16_value } -> A[ d2 ]
// CHECK-NEXT:   });
// CHECK-NEXT:   const oned_access_into_twod : mem1d_dsd = @get_dsd( mem1d_dsd, .{
// CHECK-NEXT:     .tensor_access = | d0 | { i16_value } -> B[ 1, d0 ]
// CHECK-NEXT:   });
// CHECK-NEXT:   @add16(dest_dsd, src_dsd1, src_dsd2);
// CHECK-NEXT:   @addc16(dest_dsd, i16_value, src_dsd1);
// CHECK-NEXT:   @and16(dest_dsd, u16_value, src_dsd1);
// CHECK-NEXT:   @clz(dest_dsd, i16_value);
// CHECK-NEXT:   @ctz(dest_dsd, u16_value);
// CHECK-NEXT:   @fabsh(dest_dsd, f16_value);
// CHECK-NEXT:   @fabss(dest_dsd, f32_value);
// CHECK-NEXT:   @faddh(f16_pointer, f16_value, src_dsd1);
// CHECK-NEXT:   @faddhs(f32_pointer, f32_value, src_dsd1);
// CHECK-NEXT:   @fadds(dest_dsd, src_dsd1, src_dsd2);
// CHECK-NEXT:   @fh2s(dest_dsd, f16_value);
// CHECK-NEXT:   @fh2xp16(i16_pointer, f16_value);
// CHECK-NEXT:   @fmacs(dest_dsd, src_dsd1, src_dsd2, f32_value);
// CHECK-NEXT:   @fmaxh(dest_dsd, src_dsd1, src_dsd2);
// CHECK-NEXT:   @fmaxs(dest_dsd, f32_value, src_dsd1);
// CHECK-NEXT:   @fmovh(f16_pointer, src_dsd1);
// CHECK-NEXT:   @fmovs(dest_dsd, f32_value);
// CHECK-NEXT:   @fmulh(dest_dsd, src_dsd1, src_dsd2);
// CHECK-NEXT:   @fmuls(f32_pointer, f32_value, src_dsd1);
// CHECK-NEXT:   @fnegh(dest_dsd, src_dsd1);
// CHECK-NEXT:   @fnegs(dest_dsd, f32_value);
// CHECK-NEXT:   @fnormh(f16_pointer, f16_value);
// CHECK-NEXT:   @fnorms(f32_pointer, f32_value);
// CHECK-NEXT:   @fs2h(dest_dsd, src_dsd1);
// CHECK-NEXT:   @fs2xp16(i16_pointer, f32_value);
// CHECK-NEXT:   @fscaleh(f16_pointer, f16_value, i16_value);
// CHECK-NEXT:   @fscales(f32_pointer, f32_value, i16_value);
// CHECK-NEXT:   @fsubh(f16_pointer, f16_value, src_dsd1);
// CHECK-NEXT:   @fsubs(f32_pointer, f32_value, src_dsd1);
// CHECK-NEXT:   @mov16(u16_pointer, src_dsd1);
// CHECK-NEXT:   @mov32(i32_pointer, src_dsd1);
// CHECK-NEXT:   @or16(dest_dsd, src_dsd1, u16_value);
// CHECK-NEXT:   @popcnt(dest_dsd, src_dsd1);
// CHECK-NEXT:   @sar16(dest_dsd, src_dsd1, src_dsd2);
// CHECK-NEXT:   @sll16(dest_dsd, u16_value, src_dsd1);
// CHECK-NEXT:   @slr16(dest_dsd, src_dsd1, i16_value);
// CHECK-NEXT:   @sub16(dest_dsd, src_dsd1, u16_value);
// CHECK-NEXT:   @xor16(dest_dsd, src_dsd1, src_dsd2);
// CHECK-NEXT:   @xp162fh(dest_dsd, src_dsd1);
// CHECK-NEXT:   @xp162fs(dest_dsd, src_dsd1);
// CHECK-NEXT:   @activate(@get_data_task_id(0));
// CHECK-NEXT:   @activate(@get_local_task_id(1));
// CHECK-NEXT:   @activate(@get_control_task_id(42));
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: // -----
// CHECK-NEXT: // FILE: layout.csl
// CHECK-NEXT: param param_1 : i32;
// CHECK-NEXT: param param_2 : f16 = 3.5;
// CHECK-NEXT: layout {
// CHECK-NEXT:   @set_rectangle(param_1, 6);
// CHECK-NEXT:   @set_tile_code(0, 0, "file.csl", );
// CHECK-NEXT:   const params : comptime_struct = .{
// CHECK-NEXT:     .hello = 123,
// CHECK-NEXT:   };
// CHECK-NEXT:   @set_tile_code(1, 0, "program.csl", params);
// CHECK-NEXT:   @export_name("ptr_name", [*]f32, true);
// CHECK-NEXT:   @export_name("another_ptr", [*]const i32, false);
// CHECK-NEXT:   @export_name("no_args_no_return", fn() void, );
// CHECK-NEXT:   @export_name("args_no_return", fn(i32, i32) void, );
// CHECK-NEXT: }
