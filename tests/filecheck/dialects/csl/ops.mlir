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
    %dsd_1d3 = "csl.increment_dsd_offset"(%dsd_1d2, %int16) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
    %dsd_1d4 = "csl.set_dsd_length"(%dsd_1d3, %u16) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
    %dsd_1d5 = "csl.set_dsd_stride"(%dsd_1d4, %int8) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>

    %tensor_dsd1 = "csl.get_mem_dsd"(%tens, %scalar) : (tensor<510xf32>, i32) -> !csl<dsd mem1d_dsd>
    %tensor_dsd2 = "csl.set_dsd_base_addr"(%dsd_1d, %tens) : (!csl<dsd mem1d_dsd>, tensor<510xf32>) -> !csl<dsd mem1d_dsd>

    %fabin_dsd = "csl.get_fab_dsd"(%col_1, %scalar) : (!csl.color, i32) -> !csl<dsd fabin_dsd>
    %fabout_dsd = "csl.get_fab_dsd"(%col_1, %scalar) <{"control"= true, "wavelet_index_offset" = false}>: (!csl.color, i32) -> !csl<dsd fabout_dsd>

    %f16_ptr, %f16_val, %f32_ptr = "test.op"() : () -> (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>)
    "csl.faddh"(%dsd_1d1, %dsd_1d2, %dsd_1d3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
    "csl.faddh"(%f16_ptr, %f16_val, %dsd_1d3) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
    // this will fail as expected:
    // "csl.faddh"(%f32_ptr, %f16_val, %dsd_1d3)  : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()

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
// CHECK-NEXT:     %dsd_1d3 = "csl.increment_dsd_offset"(%dsd_1d2, %int16) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %dsd_1d4 = "csl.set_dsd_length"(%dsd_1d3, %u16) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %dsd_1d5 = "csl.set_dsd_stride"(%dsd_1d4, %int8) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %tensor_dsd1 = "csl.get_mem_dsd"(%tens, %scalar) : (tensor<510xf32>, i32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %tensor_dsd2 = "csl.set_dsd_base_addr"(%dsd_1d, %tens) : (!csl<dsd mem1d_dsd>, tensor<510xf32>) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %fabin_dsd = "csl.get_fab_dsd"(%scalar) : (i32) -> !csl<dsd fabin_dsd>
// CHECK-NEXT:     %fabout_dsd = "csl.get_fab_dsd"(%scalar) : (i32) -> !csl<dsd fabout_dsd>
// CHECK-NEXT:     %f16_ptr, %f16_val, %f32_ptr = "test.op"() : () -> (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-NEXT:     "csl.faddh"(%dsd_1d1, %dsd_1d2, %dsd_1d3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.faddh"(%f16_ptr, %f16_val, %dsd_1d3) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT: csl.func @builtins() {
// CHECK-NEXT:     %i16_value, %i32_value, %u16_value, %u32_value, %f16_value, %f32_value = "test.op"() : () -> (si16, si32, ui16, ui32, f16, f32)
// CHECK-NEXT:     %i16_pointer, %i32_pointer = "test.op"() : () -> (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-NEXT:     %u16_pointer, %u32_pointer = "test.op"() : () -> (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-NEXT:     %f16_pointer, %f32_pointer = "test.op"() : () -> (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>)
// CHECK-NEXT:     %tens_1 = "test.op"() : () -> tensor<510xf32>
// CHECK-NEXT:     %dest_dsd = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %src_dsd1 = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %src_dsd2 = "csl.get_mem_dsd"(%tens_1, %i32_value) : (tensor<510xf32>, si32) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     "csl.add16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.add16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.add16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.add16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.add16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.addc16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.addc16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.addc16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.addc16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.addc16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.and16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.and16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.and16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.and16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.and16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.clz"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.clz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.clz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.ctz"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.ctz"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.ctz"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.fabsh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fabsh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fabss"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fabss"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.faddh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.faddh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.faddh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.faddh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.faddhs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.faddhs"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.faddhs"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.faddhs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fadds"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fadds"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fadds"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fadds"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fh2s"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fh2s"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fh2xp16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fh2xp16"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fh2xp16"(%i16_pointer, %f16_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
// CHECK-NEXT:     "csl.fmach"(%dest_dsd, %src_dsd1, %src_dsd2, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fmachs"(%dest_dsd, %src_dsd1, %src_dsd2, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fmacs"(%dest_dsd, %src_dsd1, %src_dsd2, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fmaxh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmaxh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmaxh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fmaxh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmaxs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmaxs"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmaxs"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fmaxs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmovh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmovh"(%f16_pointer, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmovh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fmovs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmovs"(%f32_pointer, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmovs"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fmulh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmulh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmulh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fmulh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmuls"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmuls"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fmuls"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fmuls"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fnegh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fnegh"(%dest_dsd, %f16_value) : (!csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fnegs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fnegs"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fnormh"(%f16_pointer, %f16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16) -> ()
// CHECK-NEXT:     "csl.fnorms"(%f32_pointer, %f32_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()
// CHECK-NEXT:     "csl.fs2h"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fs2h"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fs2xp16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fs2xp16"(%dest_dsd, %f32_value) : (!csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fs2xp16"(%i16_pointer, %f32_value) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, f32) -> ()
// CHECK-NEXT:     "csl.fscaleh"(%f16_pointer, %f16_value, %i16_value) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, si16) -> ()
// CHECK-NEXT:     "csl.fscales"(%f32_pointer, %f32_value, %i16_value) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, si16) -> ()
// CHECK-NEXT:     "csl.fsubh"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fsubh"(%dest_dsd, %f16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fsubh"(%dest_dsd, %src_dsd1, %f16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f16) -> ()
// CHECK-NEXT:     "csl.fsubh"(%f16_pointer, %f16_value, %src_dsd1) : (!csl.ptr<f16, #csl<ptr_kind single>, #csl<ptr_const var>>, f16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fsubs"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fsubs"(%dest_dsd, %f32_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.fsubs"(%dest_dsd, %src_dsd1, %f32_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:     "csl.fsubs"(%f32_pointer, %f32_value, %src_dsd1) : (!csl.ptr<f32, #csl<ptr_kind single>, #csl<ptr_const var>>, f32, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.mov16"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.mov16"(%i16_pointer, %src_dsd1) : (!csl.ptr<si16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.mov16"(%u16_pointer, %src_dsd1) : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.mov16"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.mov16"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.mov32"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.mov32"(%i32_pointer, %src_dsd1) : (!csl.ptr<si32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.mov32"(%u32_pointer, %src_dsd1) : (!csl.ptr<ui32, #csl<ptr_kind single>, #csl<ptr_const var>>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.mov32"(%dest_dsd, %i32_value) : (!csl<dsd mem1d_dsd>, si32) -> ()
// CHECK-NEXT:     "csl.mov32"(%dest_dsd, %u32_value) : (!csl<dsd mem1d_dsd>, ui32) -> ()
// CHECK-NEXT:     "csl.or16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.or16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.or16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.or16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.or16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.popcnt"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.popcnt"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.popcnt"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.sar16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.sar16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.sar16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.sar16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.sar16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.sll16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.sll16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.sll16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.sll16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.sll16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.slr16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.slr16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.slr16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.slr16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.slr16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.sub16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.sub16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.sub16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.xor16"(%dest_dsd, %src_dsd1, %src_dsd2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.xor16"(%dest_dsd, %i16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, si16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.xor16"(%dest_dsd, %u16_value, %src_dsd1) : (!csl<dsd mem1d_dsd>, ui16, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.xor16"(%dest_dsd, %src_dsd1, %i16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.xor16"(%dest_dsd, %src_dsd1, %u16_value) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.xp162fh"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.xp162fh"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.xp162fh"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
// CHECK-NEXT:     "csl.xp162fs"(%dest_dsd, %src_dsd1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:     "csl.xp162fs"(%dest_dsd, %i16_value) : (!csl<dsd mem1d_dsd>, si16) -> ()
// CHECK-NEXT:     "csl.xp162fs"(%dest_dsd, %u16_value) : (!csl<dsd mem1d_dsd>, ui16) -> ()
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
