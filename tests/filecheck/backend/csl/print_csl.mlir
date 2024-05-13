// RUN: xdsl-opt -t csl %s | filecheck  %s --match-full-lines

module {

csl.module {kind = #csl<module_kind layout>} {


  %p1 = "csl.param"() <{param_name = "param_1"}> : () -> i32
  %p2 = "csl.param"() <{param_name = "param_2", init_value = 1.3 : f16}> : () -> f16

  csl.layout {
    %x_dim = arith.constant 4 : i32
    %y_dim = arith.constant 6 : i32
    "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()

    %x_coord_0 = arith.constant 0 : i32
    %y_coord = arith.constant 0 : i32
    "csl.set_tile_code"(%x_coord_0, %y_coord) <{file = "file.csl"}> : (i32, i32) -> ()

    %params = "csl.const_struct"(){items = {hello = 123 : i32}} : () -> !csl.comptime_struct
    %x_coord_1 = arith.constant 1 : i32
    "csl.set_tile_code"(%x_coord_1, %y_coord, %params) <{file = "file.csl"}> : (i32, i32, !csl.comptime_struct) -> ()

  }

}

csl.module {kind = #csl<module_kind program>} {
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
  %val = arith.constant 27 : i16
  %ssa_struct = "csl.const_struct"(%val) <{ssa_fields = ["val"]}> : (i16) -> !csl.comptime_struct

  %no_param_import = "csl.import_module"() <{module = "<mod>"}> : () -> !csl.comptime_struct
  %param_import = "csl.import_module"(%ssa_struct) <{module = "<mod>"}> : (!csl.comptime_struct) -> !csl.comptime_struct

  "csl.member_call"(%param_import) <{field = "foo"}> : (!csl.comptime_struct) -> ()
  "csl.member_call"(%param_import, %val) <{field = "bar"}> : (!csl.comptime_struct, i16) -> ()
  %val2 = "csl.member_call"(%param_import) <{field = "baz"}> : (!csl.comptime_struct) -> (f32)

  %val3 = "csl.member_access"(%param_import) <{field = "f"}> :  (!csl.comptime_struct) -> (i32)


  csl.func @main() {
    "csl.call"() <{callee = @no_args_no_return}> : () -> ()
    "csl.call"(%val3, %val3) <{callee = @args_no_return}> : (i32, i32) -> ()
    %ret = "csl.call"() <{callee = @no_args_return}> : () -> (f32)

    csl.return
  }


  csl.task @data_task(%arg: f32) attributes {kind = #csl<task_kind data>, id = 0 : i5} {
    csl.return
  }

  csl.task @local_task() attributes {kind = #csl<task_kind local>, id = 1 : i5} {
    csl.return
  }

  // TODO(dk949): control_task


  "memref.global"() {"sym_name" = "uninit_array", "type" = memref<10xf32>, "sym_visibility" = "public", "initial_value"} : () -> ()
  "memref.global"() {"sym_name" = "global_array", "type" = memref<10xf32>, "sym_visibility" = "public", "initial_value" = dense<4.2> : tensor<1xf32>} : () -> ()
  "memref.global"() {"sym_name" = "const_array", "type" = memref<10xi32>, "sym_visibility" = "public", "constant", "initial_value" = dense<10> : tensor<1xi32>} : () -> ()


  %uninit_array = memref.get_global @uninit_array : memref<10xf32>
  %global_array = memref.get_global @global_array : memref<10xf32>
  %const_array = memref.get_global @const_array : memref<10xi32>

  %uninit_ptr = "csl.addressof"(%uninit_array) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const mut>>
  %global_ptr = "csl.addressof"(%global_array) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const mut>>
  %const_ptr  = "csl.addressof"(%const_array) : (memref<10xi32>) -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>

  %ptr_to_arr = "csl.addressof"(%uninit_array) : (memref<10xf32>) -> !csl.ptr<memref<10xf32>, #csl<ptr_kind single>, #csl<ptr_const mut>>
  %ptr_to_val = "csl.addressof"(%val) : (i16) -> !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const const>>


  "csl.export"(%global_ptr) <{
    type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const mut>>,
    var_name = "ptr_name"
  }> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const mut>>) -> ()

  "csl.export"(%const_ptr) <{
    type = !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>,
    var_name = "another_ptr"
  }> : (!csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>) -> ()

  "csl.export"() <{type = () -> (), sym_name = @no_args_no_return, var_name = "func_name"}> : () -> ()

  "csl.export"() <{type = (i32, i32) -> (), sym_name = @args_no_return}> : () -> ()

    %str = csl.const_str "hello"

    %ty = csl.const_type i32

    %col  = "csl.get_color"() <{id = 15 : i5}> : () -> !csl.color

    "csl.rpc"(%col) : (!csl.color) -> ()
}

}
// CHECK-NEXT: fn no_args_no_return() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }

// CHECK-NEXT: fn no_args_return() f32 {
// CHECK-NEXT:   const c : f32 = 5.0;
// CHECK-NEXT:   return c;
// CHECK-NEXT: }

// CHECK-NEXT: fn args_no_return(a : i32, b : i32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }

// CHECK-NEXT: const empty_struct : comptime_struct = .{
// CHECK-NEXT: };

// CHECK-NEXT: const attribute_struct : comptime_struct = .{
// CHECK-NEXT:   .hello = 123.0,
// CHECK-NEXT: };

// CHECK-NEXT: const val : i16 = 27;
// CHECK-NEXT: const ssa_struct : comptime_struct = .{
// CHECK-NEXT:   .val = val,
// CHECK-NEXT: };

// CHECK-NEXT: const no_param_import : imported_module = @import_module("<mod>");
// CHECK-NEXT: const param_import : imported_module = @import_module("<mod>", ssa_struct);

// CHECK-NEXT: param_import.foo();
// CHECK-NEXT: param_import.bar(val);
// CHECK-NEXT: const val2 : f32 = param_import.baz();

// CHECK-NEXT: const val3 : i32 = param_import.f;

// CHECK-NEXT: fn main() void {
// CHECK-NEXT:   no_args_no_return();
// CHECK-NEXT:   args_no_return(val3, val3);
// CHECK-NEXT:   const ret : f32 = no_args_return();
// CHECK-NEXT:   return;
// CHECK-NEXT: }

// CHECK-NEXT: task data_task(arg : f32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_data_task(data_task, @get_data_task_id(@get_color(0)));
// CHECK-NEXT: }

// CHECK-NEXT: task local_task() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_local_task(local_task, @get_local_task_id(1));
// CHECK-NEXT: }

// CHECK-NEXT: var uninit_array : [10]f32;
// CHECK-NEXT: var global_array : [10]f32 = @constants([10]f32, 4.2);
// CHECK-NEXT: const const_array : [10]i32 = @constants([10]i32, 10);

// CHECK-NEXT: var uninit_ptr : [*]f32 = &uninit_array;
// CHECK-NEXT: var global_ptr : [*]f32 = &global_array;
// CHECK-NEXT: const const_ptr : [*]const i32 = &const_array;
// CHECK-NEXT: var ptr_to_arr : *[10]f32 = &uninit_array;
// CHECK-NEXT: const ptr_to_val : *const i16 = &val;

// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(global_ptr, "ptr_name");
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(const_ptr, "another_ptr");
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(no_args_no_return, "func_name");
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(args_no_return, "args_no_return");
// CHECK-NEXT: }

// CHECK-NEXT: const str : comptime_string = "hello";

// CHECK-NEXT: const ty : type = i32;

// CHECK-NEXT: const col : color = @get_color(15);

// CHECK-NEXT: comptime {
// CHECK-NEXT:   @rpc(@get_data_task_id(col));
// CHECK-NEXT: }


// CHECK-NEXT: // >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<< //


// CHECK-NEXT: param param_1 : i32;
// CHECK-NEXT: param param_2 : f16 = 1.3;

// CHECK-NEXT: layout {

// CHECK-NEXT:   const x_dim : i32 = 4;
// CHECK-NEXT:   const y_dim : i32 = 6;
// CHECK-NEXT:   @set_rectangle(x_dim, y_dim);

// CHECK-NEXT:   const x_coord_0 : i32 = 0;
// CHECK-NEXT:   const y_coord : i32 = 0;
// CHECK-NEXT:   @set_tile_code(x_coord_0, y_coord, "file.csl", );

// CHECK-NEXT:   const params : comptime_struct = .{
// CHECK-NEXT:     .hello = 123,
// CHECK-NEXT:   };
// CHECK-NEXT:   const x_coord_1 : i32 = 1;
// CHECK-NEXT:   @set_tile_code(x_coord_1, y_coord, "file.csl", params);

// CHECK-NEXT:   @export_name("ptr_name", [*]f32, true);
// CHECK-NEXT:   @export_name("another_ptr", [*]const i32, false);
// CHECK-NEXT:   @export_name("func_name", fn() void, );
// CHECK-NEXT:   @export_name("args_no_return", fn(i32, i32) void, );

// CHECK-NEXT: }

// CHECK-EMPTY:
