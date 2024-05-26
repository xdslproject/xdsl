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

  csl.return
}
}) {sym_name = "program"} :  () -> ()

"csl.module"() <{kind = #csl<module_kind layout>}> ({
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
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT: }) {"sym_name" = "program"} :  () -> ()
// CHECK-NEXT: "csl.module"() <{"kind" = #csl<module_kind layout>}> ({
// CHECK-NEXT: csl.layout {
// CHECK-NEXT:   x_dim, %y_dim = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:   "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()
// CHECK-NEXT:   %x_coord, %y_coord, %params = "test.op"() : () -> (i32, i32, !csl.comptime_struct)
// CHECK-NEXT:   "csl.set_tile_code"(%x_coord, %y_coord, %params) <{"file" = "pe_program.csl"}> : (i32, i32, !csl.comptime_struct) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: }) {"sym_name" = "layout"} : () -> ()
// CHECK-NEXT: }
