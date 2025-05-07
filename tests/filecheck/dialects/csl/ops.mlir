// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

"csl.module"() <{kind = #csl<module_kind program>}> ({

%thing = "csl.import_module"() <{module = "<thing>"}> : () -> !csl.imported_module

csl.func @func_with_args(%arg1: i32, %arg2: i16) -> i32 {

  csl.return %arg1 : i32
}


%zero = arith.constant 0 : i32
%c100 = arith.constant 100 : i32

%zeros = "csl.constants"(%zero, %c100) : (i32, i32) -> memref<?xi32>
%zeros2 = "csl.constants"(%zero, %c100) <{is_const}> : (i32, i32) -> memref<?xi32>

csl.task @local_task() attributes {kind = #csl<task_kind local>, id = 0 : ui5} {
  csl.return
}
csl.task @data_task(%a: i32) attributes {kind = #csl<task_kind data>, id = 1 : ui5} {
  csl.return
}
csl.task @control_task() attributes {kind = #csl<task_kind control>, id = 2 : ui6} {
  csl.return
}
csl.task @control_task_args(%a: i32) attributes {kind = #csl<task_kind control>, id = 2 : ui6} {
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

    %concat = "csl.concat_structs"(%attr_struct, %ssa_struct) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct

    %three = arith.constant 3 : i16
    %col_1 = "csl.get_color"(%three) : (i16) -> !csl.color

    %three_ui = csl.mlir.signedness_cast %three : i16 to ui16

    %arr, %scalar, %tens = "test.op"() : () -> (memref<10xf32>, i32, tensor<510xf32>)
    %int8, %int16, %u16 = "test.op"() : () -> (i8, i16, i16)

    %scalar_ptr = "csl.addressof"(%scalar) : (i32) -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
    %many_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>
    %single_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<memref<10xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>

    %ptrcast = "csl.ptrcast"(%scalar_ptr) : (!csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>) -> !csl.ptr<memref<3xi32>, #csl<ptr_kind single>, #csl<ptr_const const>>
    %ptrcast_many = "csl.ptrcast"(%many_arr_ptr) : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> !csl.ptr<memref<5xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>

    %function_ptr = "csl.addressof_fn"() <{fn_name = @initialize}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
    %dir = "csl.get_dir"() <{"dir" = #csl<dir_kind north>}> : () -> !csl.direction

    %dsd_1d = "csl.get_mem_dsd"(%arr, %scalar) : (memref<10xf32>, i32) -> !csl<dsd mem1d_dsd>
    %dsd_2d = "csl.get_mem_dsd"(%arr, %scalar, %scalar) <{"tensor_access" = affine_map<(d0, d1) -> (((d0 * 3) + 1), ((d1 * 4) + 2))>}> : (memref<10xf32>, i32, i32) -> !csl<dsd mem4d_dsd>
    %dsd_3d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32) -> !csl<dsd mem4d_dsd>
    %dsd_4d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32, i32) -> !csl<dsd mem4d_dsd>
    %dsd_1d1 = "csl.set_dsd_base_addr"(%dsd_1d, %many_arr_ptr) : (!csl<dsd mem1d_dsd>, !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> !csl<dsd mem1d_dsd>
    %dsd_1d2 = "csl.set_dsd_base_addr"(%dsd_1d, %arr) : (!csl<dsd mem1d_dsd>, memref<10xf32>) -> !csl<dsd mem1d_dsd>
    %dsd_1d3 = "csl.increment_dsd_offset"(%dsd_1d2, %int16) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
    %dsd_1d4 = "csl.set_dsd_length"(%dsd_1d3, %u16) : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
    %dsd_1d5 = "csl.set_dsd_stride"(%dsd_1d4, %int8) : (!csl<dsd mem1d_dsd>, i8) -> !csl<dsd mem1d_dsd>

    %tensor_dsd1 = "csl.get_mem_dsd"(%tens, %scalar) : (tensor<510xf32>, i32) -> !csl<dsd mem1d_dsd>
    %tensor_dsd2 = "csl.set_dsd_base_addr"(%dsd_1d, %tens) : (!csl<dsd mem1d_dsd>, tensor<510xf32>) -> !csl<dsd mem1d_dsd>

    %fabin_dsd = "csl.get_fab_dsd"(%scalar) <{"fabric_color" = 2 : ui5 , "queue_id" = 0 : i3}> : (i32) -> !csl<dsd fabin_dsd>
    %fabout_dsd = "csl.get_fab_dsd"(%scalar) <{"fabric_color" = 3 : ui5 , "queue_id" = 1 : i3, "control"= true, "wavelet_index_offset" = false}>: (i32) -> !csl<dsd fabout_dsd>

    %f16_ptr, %f16_val, %f32_ptr = "test.op"() : () -> (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>)
    "csl.faddh"(%dsd_1d1, %dsd_1d2, %dsd_1d3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.faddh"(%f16_ptr, %f16_val, %dsd_1d3) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
    // this will fail as expected:
    // "csl.faddh"(%f32_ptr, %f16_val, %dsd_1d3)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()


    %one = "test.op"() : () -> i32
    %variable_with_default = "csl.variable"() <{default = 42 : i32}> : () -> !csl.var<i32>
    %variable = "csl.variable"() : () -> !csl.var<i32>
    %value = "csl.load_var"(%variable_with_default) : (!csl.var<i32>) -> i32
    %new_value = arith.addi %value, %one : i32
    "csl.store_var"(%variable_with_default, %new_value) : (!csl.var<i32>, i32) -> ()
    "csl.store_var"(%variable, %new_value) : (!csl.var<i32>, i32) -> ()

  csl.return
}

csl.func @builtins() {
    %i16_value, %i32_value, %u16_value, %u32_value, %f16_value, %f32_value = "test.op"() : () -> (si16, si32, ui16, ui32, f16, f32)
    %i16_pointer, %i32_pointer = "test.op"() : () -> (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>)
    %u16_pointer, %u32_pointer = "test.op"() : () -> (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>)
    %f16_pointer, %f32_pointer = "test.op"() : () -> (!csl.ptr<f16,  #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<f32,  #csl<ptr_kind single>, #csl<ptr_const var>>)
    %tens = "test.op"() : () -> (tensor<510xf32>)
    %dest_dsd = "csl.get_mem_dsd"(%tens, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
    %src_dsd1 = "csl.get_mem_dsd"(%tens, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
    %src_dsd2 = "csl.get_mem_dsd"(%tens, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>

    "csl.add16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.add16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
    "csl.add16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
    "csl.add16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.add16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.addc16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.addc16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
    "csl.addc16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
    "csl.addc16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.addc16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.and16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.and16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
    "csl.and16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
    "csl.and16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.and16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.clz"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.clz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
    "csl.clz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.ctz"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.ctz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
    "csl.ctz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.fabsh"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fabsh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()

    "csl.fabss"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fabss"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()

    "csl.faddh"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.faddh"(%dest_dsd,    %f16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
    "csl.faddh"(%dest_dsd,    %src_dsd1,  %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
    "csl.faddh"(%f16_pointer, %f16_value, %src_dsd1)  : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()

    "csl.faddhs"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.faddhs"(%dest_dsd,    %f16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
    "csl.faddhs"(%dest_dsd,    %src_dsd1,  %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
    "csl.faddhs"(%f32_pointer, %f32_value, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()

    "csl.fadds"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fadds"(%dest_dsd,    %f32_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
    "csl.fadds"(%dest_dsd,    %src_dsd1,  %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
    "csl.fadds"(%f32_pointer, %f32_value, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()

    "csl.fh2s"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fh2s"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()

    "csl.fh2xp16"(%dest_dsd,    %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fh2xp16"(%dest_dsd,    %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
    "csl.fh2xp16"(%i16_pointer, %f16_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()

    "csl.fmach" (%dest_dsd, %src_dsd1, %src_dsd2, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
    "csl.fmachs"(%dest_dsd, %src_dsd1, %src_dsd2, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
    "csl.fmacs" (%dest_dsd, %src_dsd1, %src_dsd2, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()

    "csl.fmaxh"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmaxh"(%dest_dsd,    %f16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmaxh"(%dest_dsd,    %src_dsd1,  %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
    "csl.fmaxh"(%f16_pointer, %f16_value, %src_dsd1)  : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()

    "csl.fmaxs"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmaxs"(%dest_dsd,    %f32_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmaxs"(%dest_dsd,    %src_dsd1,  %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
    "csl.fmaxs"(%f32_pointer, %f32_value, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()

    "csl.fmovh"(%dest_dsd,    %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmovh"(%f16_pointer, %src_dsd1)  : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmovh"(%dest_dsd,    %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()

    "csl.fmovs"(%dest_dsd,    %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmovs"(%f32_pointer, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmovs"(%dest_dsd,    %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()

    "csl.fmulh"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmulh"(%dest_dsd,    %f16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmulh"(%dest_dsd,    %src_dsd1,  %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
    "csl.fmulh"(%f16_pointer, %f16_value, %src_dsd1)  : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()

    "csl.fmuls"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmuls"(%dest_dsd,    %f32_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
    "csl.fmuls"(%dest_dsd,    %src_dsd1,  %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
    "csl.fmuls"(%f32_pointer, %f32_value, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()

    "csl.fnegh"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fnegh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()

    "csl.fnegs"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fnegs"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()

    "csl.fnormh"(%f16_pointer, %f16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
    "csl.fnorms"(%f32_pointer, %f32_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()

    "csl.fs2h"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fs2h"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()

    "csl.fs2xp16"(%dest_dsd,    %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fs2xp16"(%dest_dsd,    %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
    "csl.fs2xp16"(%i16_pointer, %f32_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()

    "csl.fscaleh"(%f16_pointer, %f16_value, %i16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, si16) -> ()
    "csl.fscales"(%f32_pointer, %f32_value, %i16_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, si16) -> ()

    "csl.fsubh"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fsubh"(%dest_dsd,    %f16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
    "csl.fsubh"(%dest_dsd,    %src_dsd1,  %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
    "csl.fsubh"(%f16_pointer, %f16_value, %src_dsd1)  : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()

    "csl.fsubs"(%dest_dsd,    %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.fsubs"(%dest_dsd,    %f32_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
    "csl.fsubs"(%dest_dsd,    %src_dsd1,  %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
    "csl.fsubs"(%f32_pointer, %f32_value, %src_dsd1)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()

    "csl.mov16"(%dest_dsd,    %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.mov16"(%i16_pointer, %src_dsd1)  : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
    "csl.mov16"(%u16_pointer, %src_dsd1)  : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
    "csl.mov16"(%dest_dsd,    %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
    "csl.mov16"(%dest_dsd,    %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.mov32"(%dest_dsd,    %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.mov32"(%i32_pointer, %src_dsd1)  : (!csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
    "csl.mov32"(%u32_pointer, %src_dsd1)  : (!csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
    "csl.mov32"(%dest_dsd,    %i32_value) : (!csl<dsd mem1d_dsd>, si32) -> ()
    "csl.mov32"(%dest_dsd,    %u32_value) : (!csl<dsd mem1d_dsd>, ui32) -> ()

    "csl.or16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.or16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
    "csl.or16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
    "csl.or16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.or16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.popcnt"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.popcnt"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
    "csl.popcnt"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.sar16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.sar16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
    "csl.sar16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
    "csl.sar16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.sar16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.sll16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.sll16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
    "csl.sll16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
    "csl.sll16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.sll16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.slr16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.slr16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
    "csl.slr16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
    "csl.slr16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.slr16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.sub16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.sub16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.sub16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.xor16"(%dest_dsd, %src_dsd1,  %src_dsd2)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.xor16"(%dest_dsd, %i16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
    "csl.xor16"(%dest_dsd, %u16_value, %src_dsd1)  : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
    "csl.xor16"(%dest_dsd, %src_dsd1,  %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
    "csl.xor16"(%dest_dsd, %src_dsd1,  %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.xp162fh"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.xp162fh"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
    "csl.xp162fh"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()

    "csl.xp162fs"(%dest_dsd, %src_dsd1)  : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.xp162fs"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
    "csl.xp162fs"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()

    csl.activate local, 0 : ui6

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
  %init = arith.constant 3.14 : f16
  %p2 = "csl.param"(%init) <{param_name = "param_2"}> : (f16) -> f16
  csl.layout {
    %x_dim, %y_dim = "test.op"() : () -> (i32, i32)
    "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()


    %x_coord, %y_coord, %params = "test.op"() : () -> (i32, i32, !csl.comptime_struct)
    "csl.set_tile_code"(%x_coord, %y_coord, %params) <{file = "pe_program.csl"}> : (i32, i32, !csl.comptime_struct) -> ()
  }
}) {sym_name = "layout"} : () -> ()


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl.module"() <{kind = #csl<module_kind program>}> ({
// CHECK-NEXT:     %thing = "csl.import_module"() <{module = "<thing>"}> : () -> !csl.imported_module
// CHECK-NEXT:     csl.func @func_with_args(%arg1 : i32, %arg2 : i16) -> i32 {
// CHECK-NEXT:       csl.return %arg1 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %zero = arith.constant 0 : i32
// CHECK-NEXT:     %c100 = arith.constant 100 : i32
// CHECK-NEXT:     %zeros = "csl.constants"(%zero, %c100) : (i32, i32) -> memref<?xi32>
// CHECK-NEXT:     %zeros2 = "csl.constants"(%zero, %c100) <{is_const}> : (i32, i32) -> memref<?xi32>
// CHECK-NEXT:     csl.task @local_task() attributes {kind = #csl<task_kind local>, id = 0 : ui5} {
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.task @data_task(%a : i32)  attributes {kind = #csl<task_kind data>, id = 1 : ui5} {
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.task @control_task() attributes {kind = #csl<task_kind control>, id = 2 : ui6} {
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.task @control_task_args(%a_1 : i32)  attributes {kind = #csl<task_kind control>, id = 2 : ui6} {
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.task @runtime_bound_local_task() attributes {kind = #csl<task_kind local>} {
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @initialize() {
// CHECK-NEXT:       %lb, %ub = "test.op"() : () -> (i16, i16)
// CHECK-NEXT:       "csl.member_call"(%thing, %lb, %ub) <{field = "some_func"}> : (!csl.imported_module, i16, i16) -> ()
// CHECK-NEXT:       %res = "csl.member_call"(%thing, %lb, %ub) <{field = "some_func"}> : (!csl.imported_module, i16, i16) -> i32
// CHECK-NEXT:       %0 = "csl.member_access"(%thing) <{field = "some_field"}> : (!csl.imported_module) -> !csl.comptime_struct
// CHECK-NEXT:       %single_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %single_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:       %many_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>
// CHECK-NEXT:       %many_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:       %col = "test.op"() : () -> !csl.color
// CHECK-NEXT:       %arg1_1, %arg2_1 = "test.op"() : () -> (i32, i16)
// CHECK-NEXT:       %call_res = "csl.call"(%arg1_1, %arg2_1) <{callee = @func_with_args}> : (i32, i16) -> i32
// CHECK-NEXT:       %attr_struct = "csl.const_struct"() <{items = {i = 42 : i32, f = 3.700000e+00 : f32}}> : () -> !csl.comptime_struct
// CHECK-NEXT:       %ssa_struct = "csl.const_struct"(%arg1_1, %arg2_1, %col) <{ssa_fields = ["i32_", "i16_", "col"]}> : (i32, i16, !csl.color) -> !csl.comptime_struct
// CHECK-NEXT:       %mixed_struct = "csl.const_struct"(%arg1_1, %arg2_1, %col) <{ssa_fields = ["i32_", "i16_", "col"], items = {i = 42 : i32, f = 3.700000e+00 : f32}}> : (i32, i16, !csl.color) -> !csl.comptime_struct
// CHECK-NEXT:       %concat = "csl.concat_structs"(%attr_struct, %ssa_struct) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:       %three = arith.constant 3 : i16
// CHECK-NEXT:       %col_1 = "csl.get_color"(%three) : (i16) -> !csl.color
// CHECK-NEXT:       %three_ui = csl.mlir.signedness_cast %three : i16 to ui16
// CHECK-NEXT:       %arr, %scalar, %tens = "test.op"() : () -> (memref<10xf32>, i32, tensor<510xf32>)
// CHECK-NEXT:       %int8, %int16, %u16 = "test.op"() : () -> (i8, i16, i16)
// CHECK-NEXT:       %scalar_ptr = "csl.addressof"(%scalar) : (i32) -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %many_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>
// CHECK-NEXT:       %single_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<memref<10xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %ptrcast = "csl.ptrcast"(%scalar_ptr) : (!csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>) -> !csl.ptr<memref<3xi32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %ptrcast_many = "csl.ptrcast"(%many_arr_ptr) : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> !csl.ptr<memref<5xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %function_ptr = "csl.addressof_fn"() <{fn_name = @initialize}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %dir = "csl.get_dir"() <{dir = #csl<dir_kind north>}> : () -> !csl.direction
// CHECK-NEXT:       %dsd_1d = "csl.get_mem_dsd"(%arr, %scalar) : (memref<10xf32>, i32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %dsd_2d = "csl.get_mem_dsd"(%arr, %scalar, %scalar) <{tensor_access = affine_map<(d0, d1) -> (((d0 * 3) + 1), ((d1 * 4) + 2))>}> : (memref<10xf32>, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       %dsd_3d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       %dsd_4d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       %dsd_1d1 = "csl.set_dsd_base_addr"(%dsd_1d, %many_arr_ptr) : (!csl<dsd mem1d_dsd>, !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %dsd_1d2 = "csl.set_dsd_base_addr"(%dsd_1d, %arr) : (!csl<dsd mem1d_dsd>, memref<10xf32>) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %dsd_1d3 = "csl.increment_dsd_offset"(%dsd_1d2, %int16) <{elem_type = f32}> : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %dsd_1d4 = "csl.set_dsd_length"(%dsd_1d3, %u16) : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %dsd_1d5 = "csl.set_dsd_stride"(%dsd_1d4, %int8) : (!csl<dsd mem1d_dsd>, i8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %tensor_dsd1 = "csl.get_mem_dsd"(%tens, %scalar) : (tensor<510xf32>, i32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %tensor_dsd2 = "csl.set_dsd_base_addr"(%dsd_1d, %tens) : (!csl<dsd mem1d_dsd>, tensor<510xf32>) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %fabin_dsd = "csl.get_fab_dsd"(%scalar) <{fabric_color = 2 : ui5, queue_id = 0 : i3}> : (i32) -> !csl<dsd fabin_dsd>
// CHECK-NEXT:       %fabout_dsd = "csl.get_fab_dsd"(%scalar) <{fabric_color = 3 : ui5, queue_id = 1 : i3, control = true, wavelet_index_offset = false}> : (i32) -> !csl<dsd fabout_dsd>
// CHECK-NEXT:       %f16_ptr, %f16_val, %f32_ptr = "test.op"() : () -> (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-NEXT:       "csl.faddh"(%dsd_1d1, %dsd_1d2, %dsd_1d3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.faddh"(%f16_ptr, %f16_val, %dsd_1d3) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       %one = "test.op"() : () -> i32
// CHECK-NEXT:       %variable_with_default = "csl.variable"() <{default = 42 : i32}> : () -> !csl.var<i32>
// CHECK-NEXT:       %variable = "csl.variable"() : () -> !csl.var<i32>
// CHECK-NEXT:       %value = "csl.load_var"(%variable_with_default) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %new_value = arith.addi %value, %one : i32
// CHECK-NEXT:       "csl.store_var"(%variable_with_default, %new_value) : (!csl.var<i32>, i32) -> ()
// CHECK-NEXT:       "csl.store_var"(%variable, %new_value) : (!csl.var<i32>, i32) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @builtins() {
// CHECK-NEXT:       %i16_value, %i32_value, %u16_value, %u32_value, %f16_value, %f32_value = "test.op"() : () -> (si16, si32, ui16, ui32, f16, f32)
// CHECK-NEXT:       %i16_pointer, %i32_pointer = "test.op"() : () -> (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-NEXT:       %u16_pointer, %u32_pointer = "test.op"() : () -> (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-NEXT:       %f16_pointer, %f32_pointer = "test.op"() : () -> (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-NEXT:       %tens_1 = "test.op"() : () -> tensor<510xf32>
// CHECK-NEXT:       %dest_dsd = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %src_dsd1 = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %src_dsd2 = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       "csl.add16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.add16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.add16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.add16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.add16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.addc16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.addc16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.addc16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.addc16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.addc16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.and16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.and16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.and16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.and16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.and16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.clz"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.clz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.clz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.ctz"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.ctz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.ctz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.fabsh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fabsh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fabss"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fabss"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.faddh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.faddh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.faddh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.faddh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.faddhs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.faddhs"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.faddhs"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.faddhs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fadds"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fadds"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fadds"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fadds"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fh2s"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fh2s"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fh2xp16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fh2xp16"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fh2xp16"(%i16_pointer, %f16_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
// CHECK-NEXT:       "csl.fmach"(%dest_dsd, %src_dsd1, %src_dsd2, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fmachs"(%dest_dsd, %src_dsd1, %src_dsd2, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fmacs"(%dest_dsd, %src_dsd1, %src_dsd2, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fmaxh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmaxh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmaxh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fmaxh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmaxs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmaxs"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmaxs"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fmaxs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmovh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmovh"(%f16_pointer, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmovh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fmovs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmovs"(%f32_pointer, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmovs"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fmulh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmulh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmulh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fmulh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmuls"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmuls"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fmuls"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fmuls"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fnegh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fnegh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fnegs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fnegs"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fnormh"(%f16_pointer, %f16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
// CHECK-NEXT:       "csl.fnorms"(%f32_pointer, %f32_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()
// CHECK-NEXT:       "csl.fs2h"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fs2h"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fs2xp16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fs2xp16"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fs2xp16"(%i16_pointer, %f32_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()
// CHECK-NEXT:       "csl.fscaleh"(%f16_pointer, %f16_value, %i16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, si16) -> ()
// CHECK-NEXT:       "csl.fscales"(%f32_pointer, %f32_value, %i16_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, si16) -> ()
// CHECK-NEXT:       "csl.fsubh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fsubh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fsubh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:       "csl.fsubh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fsubs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fsubs"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fsubs"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:       "csl.fsubs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.mov16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.mov16"(%i16_pointer, %src_dsd1) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.mov16"(%u16_pointer, %src_dsd1) : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.mov16"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.mov16"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.mov32"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.mov32"(%i32_pointer, %src_dsd1) : (!csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.mov32"(%u32_pointer, %src_dsd1) : (!csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.mov32"(%dest_dsd, %i32_value) : (!csl<dsd mem1d_dsd>, si32) -> ()
// CHECK-NEXT:       "csl.mov32"(%dest_dsd, %u32_value) : (!csl<dsd mem1d_dsd>, ui32) -> ()
// CHECK-NEXT:       "csl.or16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.or16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.or16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.or16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.or16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.popcnt"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.popcnt"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.popcnt"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.sar16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.sar16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.sar16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.sar16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.sar16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.sll16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.sll16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.sll16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.sll16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.sll16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.slr16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.slr16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.slr16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.slr16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.slr16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.sub16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.sub16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.sub16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.xor16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.xor16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.xor16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.xor16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.xor16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.xp162fh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.xp162fh"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.xp162fh"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       "csl.xp162fs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.xp162fs"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:       "csl.xp162fs"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:       csl.activate local, 0 : ui6
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     %global_ptr = "test.op"() : () -> !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"() <{var_name = @initialize, type = () -> ()}> : () -> ()
// CHECK-NEXT:     "csl.export"(%global_ptr) <{var_name = "some_name", type = !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>}> : (!csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     %rpc_col = "test.op"() : () -> !csl.color
// CHECK-NEXT:     "csl.rpc"(%rpc_col) : (!csl.color) -> ()
// CHECK-NEXT:   }) {sym_name = "program"} : () -> ()
// CHECK-NEXT:   "csl.module"() <{kind = #csl<module_kind layout>}> ({
// CHECK-NEXT:     %comp_const = "csl.param"() <{param_name = "comp_constant"}> : () -> i32
// CHECK-NEXT:     %init = arith.constant 3.140620e+00 : f16
// CHECK-NEXT:     %p2 = "csl.param"(%init) <{param_name = "param_2"}> : (f16) -> f16
// CHECK-NEXT:     csl.layout {
// CHECK-NEXT:       %x_dim, %y_dim = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:       "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()
// CHECK-NEXT:       %x_coord, %y_coord, %params = "test.op"() : () -> (i32, i32, !csl.comptime_struct)
// CHECK-NEXT:       "csl.set_tile_code"(%x_coord, %y_coord, %params) <{file = "pe_program.csl"}> : (i32, i32, !csl.comptime_struct) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {sym_name = "layout"} : () -> ()
// CHECK-NEXT: }

// CHECK-GENERIC-NEXT: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "csl.module"() <{kind = #csl<module_kind program>}> ({
// CHECK-GENERIC-NEXT:     %thing = "csl.import_module"() <{module = "<thing>"}> : () -> !csl.imported_module
// CHECK-GENERIC-NEXT:     "csl.func"() <{sym_name = "func_with_args", function_type = (i32, i16) -> i32}> ({
// CHECK-GENERIC-NEXT:     ^0(%arg1 : i32, %arg2 : i16):
// CHECK-GENERIC-NEXT:       "csl.return"(%arg1) : (i32) -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-GENERIC-NEXT:     %c100 = "arith.constant"() <{value = 100 : i32}> : () -> i32
// CHECK-GENERIC-NEXT:     %zeros = "csl.constants"(%zero, %c100) : (i32, i32) -> memref<?xi32>
// CHECK-GENERIC-NEXT:     %zeros2 = "csl.constants"(%zero, %c100) <{is_const}> : (i32, i32) -> memref<?xi32>
// CHECK-GENERIC-NEXT:     "csl.task"() <{sym_name = "local_task", function_type = () -> (), kind = #csl<task_kind local>, id = 0 : ui5}> ({
// CHECK-GENERIC-NEXT:       "csl.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     "csl.task"() <{sym_name = "data_task", function_type = (i32) -> (), kind = #csl<task_kind data>, id = 1 : ui5}> ({
// CHECK-GENERIC-NEXT:     ^1(%a : i32):
// CHECK-GENERIC-NEXT:       "csl.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     "csl.task"() <{sym_name = "control_task", function_type = () -> (), kind = #csl<task_kind control>, id = 2 : ui6}> ({
// CHECK-GENERIC-NEXT:       "csl.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     "csl.task"() <{sym_name = "control_task_args", function_type = (i32) -> (), kind = #csl<task_kind control>, id = 2 : ui6}> ({
// CHECK-GENERIC-NEXT:     ^2(%a_1 : i32):
// CHECK-GENERIC-NEXT:       "csl.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     "csl.task"() <{sym_name = "runtime_bound_local_task", function_type = () -> (), kind = #csl<task_kind local>}> ({
// CHECK-GENERIC-NEXT:       "csl.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     "csl.func"() <{sym_name = "initialize", function_type = () -> ()}> ({
// CHECK-GENERIC-NEXT:       %lb, %ub = "test.op"() : () -> (i16, i16)
// CHECK-GENERIC-NEXT:       "csl.member_call"(%thing, %lb, %ub) <{field = "some_func"}> : (!csl.imported_module, i16, i16) -> ()
// CHECK-GENERIC-NEXT:       %res = "csl.member_call"(%thing, %lb, %ub) <{field = "some_func"}> : (!csl.imported_module, i16, i16) -> i32
// CHECK-GENERIC-NEXT:       %0 = "csl.member_access"(%thing) <{field = "some_field"}> : (!csl.imported_module) -> !csl.comptime_struct
// CHECK-GENERIC-NEXT:       %single_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-GENERIC-NEXT:       %single_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-GENERIC-NEXT:       %many_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>
// CHECK-GENERIC-NEXT:       %many_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-GENERIC-NEXT:       %col = "test.op"() : () -> !csl.color
// CHECK-GENERIC-NEXT:       %arg1_1, %arg2_1 = "test.op"() : () -> (i32, i16)
// CHECK-GENERIC-NEXT:       %call_res = "csl.call"(%arg1_1, %arg2_1) <{callee = @func_with_args}> : (i32, i16) -> i32
// CHECK-GENERIC-NEXT:       %attr_struct = "csl.const_struct"() <{items = {i = 42 : i32, f = 3.700000e+00 : f32}}> : () -> !csl.comptime_struct
// CHECK-GENERIC-NEXT:       %ssa_struct = "csl.const_struct"(%arg1_1, %arg2_1, %col) <{ssa_fields = ["i32_", "i16_", "col"]}> : (i32, i16, !csl.color) -> !csl.comptime_struct
// CHECK-GENERIC-NEXT:       %mixed_struct = "csl.const_struct"(%arg1_1, %arg2_1, %col) <{ssa_fields = ["i32_", "i16_", "col"], items = {i = 42 : i32, f = 3.700000e+00 : f32}}> : (i32, i16, !csl.color) -> !csl.comptime_struct
// CHECK-GENERIC-NEXT:       %concat = "csl.concat_structs"(%attr_struct, %ssa_struct) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-GENERIC-NEXT:       %three = "arith.constant"() <{value = 3 : i16}> : () -> i16
// CHECK-GENERIC-NEXT:       %col_1 = "csl.get_color"(%three) : (i16) -> !csl.color
// CHECK-GENERIC-NEXT:       %three_ui = "csl.mlir.signedness_cast"(%three) : (i16) -> ui16
// CHECK-GENERIC-NEXT:       %arr, %scalar, %tens = "test.op"() : () -> (memref<10xf32>, i32, tensor<510xf32>)
// CHECK-GENERIC-NEXT:       %int8, %int16, %u16 = "test.op"() : () -> (i8, i16, i16)
// CHECK-GENERIC-NEXT:       %scalar_ptr = "csl.addressof"(%scalar) : (i32) -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-GENERIC-NEXT:       %many_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>
// CHECK-GENERIC-NEXT:       %single_arr_ptr = "csl.addressof"(%arr) : (memref<10xf32>) -> !csl.ptr<memref<10xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-GENERIC-NEXT:       %ptrcast = "csl.ptrcast"(%scalar_ptr) : (!csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>) -> !csl.ptr<memref<3xi32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-GENERIC-NEXT:       %ptrcast_many = "csl.ptrcast"(%many_arr_ptr) : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> !csl.ptr<memref<5xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-GENERIC-NEXT:       %function_ptr = "csl.addressof_fn"() <{fn_name = @initialize}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-GENERIC-NEXT:       %dir = "csl.get_dir"() <{dir = #csl<dir_kind north>}> : () -> !csl.direction
// CHECK-GENERIC-NEXT:       %dsd_1d = "csl.get_mem_dsd"(%arr, %scalar) : (memref<10xf32>, i32) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %dsd_2d = "csl.get_mem_dsd"(%arr, %scalar, %scalar) <{tensor_access = affine_map<(d0, d1) -> (((d0 * 3) + 1), ((d1 * 4) + 2))>}> : (memref<10xf32>, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-GENERIC-NEXT:       %dsd_3d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-GENERIC-NEXT:       %dsd_4d = "csl.get_mem_dsd"(%arr, %scalar, %scalar, %scalar, %scalar) : (memref<10xf32>, i32, i32, i32, i32) -> !csl<dsd mem4d_dsd>
// CHECK-GENERIC-NEXT:       %dsd_1d1 = "csl.set_dsd_base_addr"(%dsd_1d, %many_arr_ptr) : (!csl<dsd mem1d_dsd>, !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %dsd_1d2 = "csl.set_dsd_base_addr"(%dsd_1d, %arr) : (!csl<dsd mem1d_dsd>, memref<10xf32>) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %dsd_1d3 = "csl.increment_dsd_offset"(%dsd_1d2, %int16) <{elem_type = f32}> : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %dsd_1d4 = "csl.set_dsd_length"(%dsd_1d3, %u16) : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %dsd_1d5 = "csl.set_dsd_stride"(%dsd_1d4, %int8) : (!csl<dsd mem1d_dsd>, i8) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %tensor_dsd1 = "csl.get_mem_dsd"(%tens, %scalar) : (tensor<510xf32>, i32) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %tensor_dsd2 = "csl.set_dsd_base_addr"(%dsd_1d, %tens) : (!csl<dsd mem1d_dsd>, tensor<510xf32>) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %fabin_dsd = "csl.get_fab_dsd"(%scalar) <{fabric_color = 2 : ui5, queue_id = 0 : i3}> : (i32) -> !csl<dsd fabin_dsd>
// CHECK-GENERIC-NEXT:       %fabout_dsd = "csl.get_fab_dsd"(%scalar) <{fabric_color = 3 : ui5, queue_id = 1 : i3, control = true, wavelet_index_offset = false}> : (i32) -> !csl<dsd fabout_dsd>
// CHECK-GENERIC-NEXT:       %f16_ptr, %f16_val, %f32_ptr = "test.op"() : () -> (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-GENERIC-NEXT:       "csl.faddh"(%dsd_1d1, %dsd_1d2, %dsd_1d3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddh"(%f16_ptr, %f16_val, %dsd_1d3) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       %one = "test.op"() : () -> i32
// CHECK-GENERIC-NEXT:       %variable_with_default = "csl.variable"() <{default = 42 : i32}> : () -> !csl.var<i32>
// CHECK-GENERIC-NEXT:       %variable = "csl.variable"() : () -> !csl.var<i32>
// CHECK-GENERIC-NEXT:       %value = "csl.load_var"(%variable_with_default) : (!csl.var<i32>) -> i32
// CHECK-GENERIC-NEXT:       %new_value = "arith.addi"(%value, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:       "csl.store_var"(%variable_with_default, %new_value) : (!csl.var<i32>, i32) -> ()
// CHECK-GENERIC-NEXT:       "csl.store_var"(%variable, %new_value) : (!csl.var<i32>, i32) -> ()
// CHECK-GENERIC-NEXT:       "csl.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     "csl.func"() <{sym_name = "builtins", function_type = () -> ()}> ({
// CHECK-GENERIC-NEXT:       %i16_value, %i32_value, %u16_value, %u32_value, %f16_value, %f32_value = "test.op"() : () -> (si16, si32, ui16, ui32, f16, f32)
// CHECK-GENERIC-NEXT:       %i16_pointer, %i32_pointer = "test.op"() : () -> (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-GENERIC-NEXT:       %u16_pointer, %u32_pointer = "test.op"() : () -> (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-GENERIC-NEXT:       %f16_pointer, %f32_pointer = "test.op"() : () -> (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-GENERIC-NEXT:       %tens_1 = "test.op"() : () -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %dest_dsd = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %src_dsd1 = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       %src_dsd2 = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-GENERIC-NEXT:       "csl.add16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.add16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.add16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.add16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.add16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.addc16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.addc16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.addc16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.addc16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.addc16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.and16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.and16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.and16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.and16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.and16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.clz"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.clz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.clz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.ctz"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.ctz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.ctz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fabsh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fabsh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fabss"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fabss"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddhs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddhs"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddhs"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.faddhs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fadds"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fadds"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fadds"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fadds"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fh2s"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fh2s"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fh2xp16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fh2xp16"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fh2xp16"(%i16_pointer, %f16_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmach"(%dest_dsd, %src_dsd1, %src_dsd2, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmachs"(%dest_dsd, %src_dsd1, %src_dsd2, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmacs"(%dest_dsd, %src_dsd1, %src_dsd2, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmaxh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmaxh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmaxh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmaxh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmaxs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmaxs"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmaxs"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmaxs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmovh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmovh"(%f16_pointer, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmovh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmovs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmovs"(%f32_pointer, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmovs"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmulh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmulh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmulh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmulh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmuls"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmuls"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmuls"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fmuls"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fnegh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fnegh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fnegs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fnegs"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fnormh"(%f16_pointer, %f16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fnorms"(%f32_pointer, %f32_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fs2h"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fs2h"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fs2xp16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fs2xp16"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fs2xp16"(%i16_pointer, %f32_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fscaleh"(%f16_pointer, %f16_value, %i16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fscales"(%f32_pointer, %f32_value, %i16_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fsubh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fsubh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fsubh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-GENERIC-NEXT:       "csl.fsubh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fsubs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fsubs"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.fsubs"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-GENERIC-NEXT:       "csl.fsubs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov16"(%i16_pointer, %src_dsd1) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov16"(%u16_pointer, %src_dsd1) : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov16"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov16"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov32"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov32"(%i32_pointer, %src_dsd1) : (!csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov32"(%u32_pointer, %src_dsd1) : (!csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov32"(%dest_dsd, %i32_value) : (!csl<dsd mem1d_dsd>, si32) -> ()
// CHECK-GENERIC-NEXT:       "csl.mov32"(%dest_dsd, %u32_value) : (!csl<dsd mem1d_dsd>, ui32) -> ()
// CHECK-GENERIC-NEXT:       "csl.or16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.or16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.or16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.or16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.or16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.popcnt"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.popcnt"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.popcnt"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.sar16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.sar16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.sar16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.sar16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.sar16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.sll16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.sll16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.sll16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.sll16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.sll16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.slr16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.slr16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.slr16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.slr16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.slr16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.sub16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.sub16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.sub16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.xor16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.xor16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.xor16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.xor16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.xor16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.xp162fh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.xp162fh"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.xp162fh"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.xp162fs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-GENERIC-NEXT:       "csl.xp162fs"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-GENERIC-NEXT:       "csl.xp162fs"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-GENERIC-NEXT:       "csl.activate"() <{kind = #csl<task_kind local>, id = 0 : ui6}> : () -> ()
// CHECK-GENERIC-NEXT:       "csl.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     %global_ptr = "test.op"() : () -> !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-GENERIC-NEXT:     "csl.export"() <{var_name = @initialize, type = () -> ()}> : () -> ()
// CHECK-GENERIC-NEXT:     "csl.export"(%global_ptr) <{var_name = "some_name", type = !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>}> : (!csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
// CHECK-GENERIC-NEXT:     %rpc_col = "test.op"() : () -> !csl.color
// CHECK-GENERIC-NEXT:     "csl.rpc"(%rpc_col) : (!csl.color) -> ()
// CHECK-GENERIC-NEXT:   }) {sym_name = "program"} : () -> ()
// CHECK-GENERIC-NEXT:   "csl.module"() <{kind = #csl<module_kind layout>}> ({
// CHECK-GENERIC-NEXT:     %comp_const = "csl.param"() <{param_name = "comp_constant"}> : () -> i32
// CHECK-GENERIC-NEXT:     %init = "arith.constant"() <{value = 3.140620e+00 : f16}> : () -> f16
// CHECK-GENERIC-NEXT:     %p2 = "csl.param"(%init) <{param_name = "param_2"}> : (f16) -> f16
// CHECK-GENERIC-NEXT:     "csl.layout"() ({
// CHECK-GENERIC-NEXT:       %x_dim, %y_dim = "test.op"() : () -> (i32, i32)
// CHECK-GENERIC-NEXT:       "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()
// CHECK-GENERIC-NEXT:       %x_coord, %y_coord, %params = "test.op"() : () -> (i32, i32, !csl.comptime_struct)
// CHECK-GENERIC-NEXT:       "csl.set_tile_code"(%x_coord, %y_coord, %params) <{file = "pe_program.csl"}> : (i32, i32, !csl.comptime_struct) -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:   }) {sym_name = "layout"} : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
