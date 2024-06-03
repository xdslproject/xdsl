// RUN: XDSL_ROUNDTRIP

"csl.module"() <{kind = #csl<module_kind program>}> ({

%thing = "csl.import_module"() <{module = "<thing>"}> : () -> !csl.imported_module

csl.func @func_with_args(%arg1: i32, %arg2: i16) -> i32 {

  csl.return %arg1 : i32
}


csl.task @local_task() attributes {kind = #csl<task_kind local>, id = 0 : i5} {
  csl.return
}
csl.task @data_task(%a: i32) attributes {kind = #csl<task_kind data>, id = 1 : i5} {
  csl.return
}
csl.task @control_task() attributes {kind = #csl<task_kind control>, id = 2 : i6} {
  csl.return
}
csl.task @control_task_args(%a: i32) attributes {kind = #csl<task_kind control>, id = 2 : i6} {
  csl.return
}
csl.task @runtime_bound_local_task() attributes {kind = #csl<task_kind local>} {
  csl.return
}


csl.func @initialize() {

    %lb, %ub = "test.op"() : () -> (i16, i16)

    "csl.member_call"(%thing, %lb, %ub) <{field = "some_func"}> : (!csl.imported_module, i16, i16) -> ()

    %res = "csl.member_call"(%thing, %lb, %ub) <{field = "some_func"}> : (!csl.imported_module, i16, i16) -> (i32)

    %11 = "csl.member_access"(%thing) <{field = "some_field"}> : (!csl.imported_module) -> !csl.comptime_struct

    %single_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
    %single_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>>
    %many_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>
    %many_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const var>>

    %col = "test.op"() : () -> !csl.color

    %arg1, %arg2 = "test.op"() : () -> (i32, i16)
    %call_res = "csl.call"(%arg1, %arg2) <{callee = @func_with_args}> : (i32, i16) -> i32


    %attr_struct = "csl.const_struct"() <{
      items = {i = 42 : i32, f = 3.7 : f32 }
    }> : () -> !csl.comptime_struct

    %ssa_struct = "csl.const_struct"(%arg1, %arg2, %col) <{
      ssa_fields = ["i32_", "i16_", "col"]
    }> : (i32, i16, !csl.color) -> !csl.comptime_struct

    %mixed_struct = "csl.const_struct"(%arg1, %arg2, %col) <{
      ssa_fields = ["i32_", "i16_", "col"],
      items = {i = 42 : i32, f = 3.7 : f32 }
    }> : (i32, i16, !csl.color) -> !csl.comptime_struct

    %col_1 = "csl.get_color"() <{id = 3 : i5}> : () -> !csl.color


    %arr, %scalar, %tens = "test.op"() : () -> (memref<10xf32>, i32, tensor<510xf32>)
    %int8, %int16, %u16 = "test.op"() : () -> (si8, si16, ui16)

    %scalar_ptr = "csl.addressof"(%scalar) : (i32) -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
    %many_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>
    %single_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<memref<10xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>

    %dsd_1d = "csl.get_mem_dsd"(%arr, %scalar) : (memref<10xf32>, i32) -> !csl<dsd mem1d_dsd>
    %dsd_2d = "csl.get_mem_dsd"(%arr, %scalar, %scalar) <{"strides" = [3, 4], "offsets" = [1, 2]}> : (memref<10xf32>, i32, i32) -> !csl<dsd mem4d_dsd>
    %dsd_3d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32) -> !csl<dsd mem4d_dsd>
    %dsd_4d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32, i32) -> !csl<dsd mem4d_dsd>
    %dsd_1d1 = "csl.set_dsd_base_addr"(%dsd_1d, %many_arr_ptr) : (!csl<dsd mem1d_dsd>, !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> !csl<dsd mem1d_dsd>
    %dsd_1d2 = "csl.set_dsd_base_addr"(%dsd_1d, %arr) : (!csl<dsd mem1d_dsd>, memref<10xf32>) -> !csl<dsd mem1d_dsd>
    %dsd_1d3 = "csl.increment_dsd_offset"(%dsd_1d2, %int16) : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
    %dsd_1d4 = "csl.set_dsd_length"(%dsd_1d3, %u16) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
    //%dsd_1d4 = "csl.set_dsd_length"(%u16, %dsd_1d3) : (ui16, !csl<dsd mem1d_dsd>) -> !csl<dsd mem1d_dsd>
    %dsd_1d5 = "csl.set_dsd_stride"(%dsd_1d4, %int8) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>

    %tensor_dsd1 = "csl.get_mem_dsd"(%tens, %scalar) : (tensor<510xf32>, i32) -> !csl<dsd mem1d_dsd>
    %tensor_dsd2 = "csl.set_dsd_base_addr"(%dsd_1d, %tens) : (!csl<dsd mem1d_dsd>, tensor<510xf32>) -> !csl<dsd mem1d_dsd>

    %fabin_dsd = "csl.get_fab_dsd"(%scalar) : (i32) -> !csl<dsd fabin_dsd>
    %fabout_dsd = "csl.get_fab_dsd"(%scalar) : (i32) -> !csl<dsd fabout_dsd>


  csl.return
}

%global_ptr = "test.op"() : () -> !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>

"csl.export"() <{var_name = @initialize, type = () -> ()}> : () -> ()
"csl.export"(%global_ptr) <{
  var_name = "some_name",
  type = !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>
}> : (!csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()

%rpc_col = "test.op"() : () -> !csl.color
"csl.rpc"(%rpc_col) : (!csl.color) -> ()

}) {sym_name = "program"} :  () -> ()

"csl.module"() <{kind = #csl<module_kind layout>}> ({
  %comp_const = "csl.param"() <{param_name = "comp_constant"}> : () -> i32
  %comp_const_with_def = "csl.param"() <{param_name = "comp_constant", init_value = 1 : i32}> : () -> i32
  csl.layout {
    %x_dim, %y_dim = "test.op"() : () -> (i32, i32)
    "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()


    %x_coord, %y_coord, %params = "test.op"() : () -> (i32, i32, !csl.comptime_struct)
    "csl.set_tile_code"(%x_coord, %y_coord, %params) <{file = "pe_program.csl"}> : (i32, i32, !csl.comptime_struct) -> ()
  }
}) {sym_name = "layout"} : () -> ()


// CHECK-NEXT: builtin.module {
// CHECK-NEXT: "csl.module"() <{"kind" = #csl<module_kind program>}> ({
// CHECK-NEXT: %thing = "csl.import_module"() <{"module" = "<thing>"}> : () -> !csl.imported_module
// CHECK-NEXT: csl.func @func_with_args(%arg1 : i32, %arg2 : i16) -> i32 {
// CHECK-NEXT:   csl.return %arg1 : i32
// CHECK-NEXT: }
// CHECK-NEXT: csl.task @local_task()  attributes {"kind" = #csl<task_kind local>, "id" = 0 : i5}{
// CHECK-NEXT:   csl.return
// CHECK-NEXT: }
// CHECK-NEXT: csl.task @data_task(%a : i32) attributes {"kind" = #csl<task_kind data>, "id" = 1 : i5}{
// CHECK-NEXT:   csl.return
// CHECK-NEXT: }
// CHECK-NEXT: csl.task @control_task() attributes {"kind" = #csl<task_kind control>, "id" = 2 : i6}{
// CHECK-NEXT:   csl.return
// CHECK-NEXT: }
// CHECK-NEXT: csl.task @control_task_args(%a_1 : i32) attributes {"kind" = #csl<task_kind control>, "id" = 2 : i6}{
// CHECK-NEXT:   csl.return
// CHECK-NEXT: }
// CHECK-NEXT: csl.task @runtime_bound_local_task() attributes {"kind" = #csl<task_kind local>}{
// CHECK-NEXT:   csl.return
// CHECK-NEXT: }
// CHECK-NEXT: csl.func @initialize() {
// CHECK-NEXT:     %lb, %ub = "test.op"() : () -> (i16, i16)
// CHECK-NEXT:     "csl.member_call"(%thing, %lb, %ub) <{"field" = "some_func"}> : (!csl.imported_module, i16, i16) -> ()
// CHECK-NEXT:     %res = "csl.member_call"(%thing, %lb, %ub) <{"field" = "some_func"}> : (!csl.imported_module, i16, i16) -> i32
// CHECK-NEXT:     %0 = "csl.member_access"(%thing) <{"field" = "some_field"}> : (!csl.imported_module) -> !csl.comptime_struct
// CHECK-NEXT:     %single_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:     %single_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:     %many_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>
// CHECK-NEXT:     %many_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %col = "test.op"() : () -> !csl.color
// CHECK-NEXT:     %arg1_1, %arg2_1 = "test.op"() : () -> (i32, i16)
// CHECK-NEXT:     %call_res = "csl.call"(%arg1_1, %arg2_1) <{"callee" = @func_with_args}> : (i32, i16) -> i32
// CHECK-NEXT:     %attr_struct = "csl.const_struct"() <{"items" = {"i" = 42 : i32, "f" = 3.700000e+00 : f32}}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %ssa_struct = "csl.const_struct"(%arg1_1, %arg2_1, %col) <{"ssa_fields" = ["i32_", "i16_", "col"]}> : (i32, i16, !csl.color) -> !csl.comptime_struct
// CHECK-NEXT:     %mixed_struct = "csl.const_struct"(%arg1_1, %arg2_1, %col) <{"ssa_fields" = ["i32_", "i16_", "col"], "items" = {"i" = 42 : i32, "f" = 3.700000e+00 : f32}}> : (i32, i16, !csl.color) -> !csl.comptime_struct
// CHECK-NEXT:     %col_1 = "csl.get_color"() <{"id" = 3 : i5}> : () -> !csl.color
// CHECK-NEXT:     %arr, %scalar, %tens = "test.op"() : () -> (memref<10xf32>, i32, tensor<510xf32>)
// CHECK-NEXT:     %int8, %int16, %u16 = "test.op"() : () -> (si8, si16, ui16)
// CHECK-NEXT:     %scalar_ptr = "csl.addressof"(%scalar) : (i32) -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:     %many_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>
// CHECK-NEXT:     %single_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<memref<10xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:     %dsd_1d = "csl.get_mem_dsd"(%arr, %scalar) : (memref<10xf32>, i32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %dsd_2d = "csl.get_mem_dsd"(%arr, %scalar, %scalar) <{"strides" = [3 : i64, 4 : i64], "offsets" = [1 : i64, 2 : i64]}> : (memref<10xf32>, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:     %dsd_3d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:     %dsd_4d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:     %dsd_1d1 = "csl.set_dsd_base_addr"(%dsd_1d, %many_arr_ptr) : (!csl<dsd mem1d_dsd>, !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %dsd_1d2 = "csl.set_dsd_base_addr"(%dsd_1d, %arr) : (!csl<dsd mem1d_dsd>, memref<10xf32>) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %dsd_1d3 = "csl.increment_dsd_offset"(%dsd_1d2, %int16) : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %dsd_1d4 = "csl.set_dsd_length"(%dsd_1d3, %u16) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %dsd_1d5 = "csl.set_dsd_stride"(%dsd_1d4, %int8) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %tensor_dsd1 = "csl.get_mem_dsd"(%tens, %scalar) : (tensor<510xf32>, i32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %tensor_dsd2 = "csl.set_dsd_base_addr"(%dsd_1d, %tens) : (!csl<dsd mem1d_dsd>, tensor<510xf32>) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %fabin_dsd = "csl.get_fab_dsd"(%scalar) : (i32) -> !csl<dsd fabin_dsd>
// CHECK-NEXT:     %fabout_dsd = "csl.get_fab_dsd"(%scalar) : (i32) -> !csl<dsd fabout_dsd>
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT: %global_ptr = "test.op"() : () -> !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT: "csl.export"() <{"var_name" = @initialize, "type" = () -> ()}> : () -> ()
// CHECK-NEXT: "csl.export"(%global_ptr) <{"var_name" = "some_name", "type" = !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>}> : (!csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT: %rpc_col = "test.op"() : () -> !csl.color
// CHECK-NEXT: "csl.rpc"(%rpc_col) : (!csl.color) -> ()
// CHECK-NEXT: }) {"sym_name" = "program"} :  () -> ()
// CHECK-NEXT: "csl.module"() <{"kind" = #csl<module_kind layout>}> ({
// CHECK-NEXT:  %comp_const = "csl.param"() <{"param_name" = "comp_constant"}> : () -> i32
// CHECK-NEXT:  %comp_const_with_def = "csl.param"() <{"param_name" = "comp_constant", "init_value" = 1 : i32}> : () -> i32
// CHECK-NEXT: csl.layout {
// CHECK-NEXT:   x_dim, %y_dim = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:   "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()
// CHECK-NEXT:   %x_coord, %y_coord, %params = "test.op"() : () -> (i32, i32, !csl.comptime_struct)
// CHECK-NEXT:   "csl.set_tile_code"(%x_coord, %y_coord, %params) <{"file" = "pe_program.csl"}> : (i32, i32, !csl.comptime_struct) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: }) {"sym_name" = "layout"} : () -> ()
// CHECK-NEXT: }
