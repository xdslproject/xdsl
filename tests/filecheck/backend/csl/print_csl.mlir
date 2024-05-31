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

  csl.func @casts() {
    %constI32 = arith.constant 0 : i32
    %constU16 = arith.constant 0 : ui16
    %constF32 = arith.constant 0 : f32
    %castIndex = "arith.index_cast"(%constF32) : (f32) -> index
    %castF16 = "arith.sitofp"(%constI32) : (i32) -> f16
    %castI16 = "arith.fptosi"(%constF32) : (f32) -> i16
    %castF32 = "arith.extf"(%castF16) : (f16) -> f32
    %castF16again = "arith.truncf"(%constF32) : (f32) -> f16
    %castI16again = "arith.trunci"(%constI32) : (i32) -> i16
    %castI32again = "arith.extsi"(%castI16) : (i16) -> i32
    %castU32 = "arith.extui"(%constU16)  : (ui16) -> ui32
    csl.return
  }


  csl.task @data_task(%arg: f32) attributes {kind = #csl<task_kind data>, id = 0 : i5} {
    csl.return
  }

  csl.task @local_task() attributes {kind = #csl<task_kind local>, id = 1 : i5} {
    csl.return
  }

  csl.task @control_task() attributes {kind = #csl<task_kind control>, id = 42 : i6} {
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


  "memref.global"() {"sym_name" = "uninit_array", "type" = memref<10xf32>, "sym_visibility" = "public", "initial_value"} : () -> ()
  "memref.global"() {"sym_name" = "global_array", "type" = memref<10xf32>, "sym_visibility" = "public", "initial_value" = dense<4.2> : tensor<1xf32>} : () -> ()
  "memref.global"() {"sym_name" = "const_array", "type" = memref<10xi32>, "sym_visibility" = "public", "constant", "initial_value" = dense<10> : tensor<1xi32>} : () -> ()


  %uninit_array = memref.get_global @uninit_array : memref<10xf32>
  %global_array = memref.get_global @global_array : memref<10xf32>
  %const_array = memref.get_global @const_array : memref<10xi32>

  %uninit_ptr = "csl.addressof"(%uninit_array) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %global_ptr = "csl.addressof"(%global_array) : (memref<10xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %const_ptr  = "csl.addressof"(%const_array) : (memref<10xi32>) -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>

  %ptr_to_arr = "csl.addressof"(%uninit_array) : (memref<10xf32>) -> !csl.ptr<memref<10xf32>, #csl<ptr_kind single>, #csl<ptr_const var>>
  %ptr_to_val = "csl.addressof"(%const27) : (i16) -> !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const const>>


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

    %col  = "csl.get_color"() <{id = 15 : i5}> : () -> !csl.color

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

  %0 = arith.constant 3.14 : f32
  %v0 = arith.constant 2.718 : f16

  %u32cst = arith.constant 44 : ui32

  %A = memref.get_global @A : memref<24xf32>
  %x = memref.get_global @x : memref<6xf32>
  %b = memref.get_global @b : memref<4xf32>
  %y = memref.get_global @y : memref<4xf32>

  scf.for %idx = %lb to %ub step %step {
    %idx_f32 = arith.sitofp %idx : i16 to f32
    %idx_index = "arith.index_cast"(%idx) : (i16) -> index
    memref.store %idx_f32, %A[%idx_index] : memref<24xf32>
  }

  %ub_6 = arith.constant 6 : i16

  scf.for %j = %lb to %ub_6 step %step {
    %val = arith.constant 1.0 : f32
    %j_idx = "arith.index_cast"(%j) : (i16) -> index
    memref.store %val, %x[%j_idx] : memref<6xf32>
  }

  %ub_4 = arith.constant 6 : i16

  scf.for %i = %lb to %ub_4 step %step {
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

}) {sym_name = "program"} : () -> ()


"csl.module"() <{kind=#csl<module_kind layout>}> ({
  %p1 = "csl.param"() <{param_name = "param_1"}> : () -> i32
  %p2 = "csl.param"() <{param_name = "param_2", init_value = 1.3 : f16}> : () -> f16

  csl.layout {
    %x_dim = arith.constant 4 : i32
    %y_dim = arith.constant 6 : i32
    "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()

    %x_coord0 = arith.constant 0 : i32
    %y_coord = arith.constant 0 : i32
    "csl.set_tile_code"(%x_coord0, %y_coord) <{file = "file.csl"}> : (i32, i32) -> ()

    %params = "csl.const_struct"(){items = {hello = 123 : i32}} : () -> !csl.comptime_struct
    %x_coord1 = arith.constant 1 : i32
    "csl.set_tile_code"(%x_coord1, %y_coord, %params) <{file = "file.csl"}> : (i32, i32, !csl.comptime_struct) -> ()

  }
}) {sym_name = "layout"} : () -> ()

// CHECK-NEXT:
// CHECK-NEXT: fn no_args_no_return() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: fn no_args_return() f32 {
// CHECK-NEXT:   const c : f32 = 5.0;
// CHECK-NEXT:   return c;
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: fn args_no_return(a : i32, b : i32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: //unknown op ConstStructOp(%empty_struct = "csl.const_struct"() : () -> !csl.comptime_struct)
// CHECK-NEXT: //unknown op ConstStructOp(%attribute_struct = "csl.const_struct"() <{"items" = {"hello" = 1.230000e+02 : f32}}> : () -> !csl.comptime_struct)
// CHECK-NEXT: const const27 : i16 = 27;
// CHECK-NEXT: //unknown op ConstStructOp(%ssa_struct = "csl.const_struct"(%const27) <{"ssa_fields" = ["val"]}> : (i16) -> !csl.comptime_struct)
// CHECK-NEXT: const no_param_import : imported_module = @import_module("<mod>");
// CHECK-NEXT: const param_import : imported_module = @import_module("<mod>", ssa_struct);
// CHECK-NEXT: param_import.foo();
// CHECK-NEXT: param_import.bar(const27);
// CHECK-NEXT: const val2 : f32 = param_import.baz();
// CHECK-NEXT: const val3 : i32 = param_import.f;
// CHECK-NEXT:
// CHECK-NEXT: fn main() void {
// CHECK-NEXT:   no_args_no_return();
// CHECK-NEXT:   args_no_return(val3, val3);
// CHECK-NEXT:   const ret : f32 = no_args_return();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: fn casts() void {
// CHECK-NEXT:   const constI32 : i32 = 0;
// CHECK-NEXT:   const constU16 : u16 = 0;
// CHECK-NEXT:   const constF32 : f32 = 0.0;
// CHECK-NEXT:   const castIndex : i32 = @as(i32, constF32);
// CHECK-NEXT:   const castF16 : f16 = @as(f16, constI32);
// CHECK-NEXT:   const castI16 : i16 = @as(i16, constF32);
// CHECK-NEXT:   const castF32 : f32 = @as(f32, castF16);
// CHECK-NEXT:   const castF16again : f16 = @as(f16, constF32);
// CHECK-NEXT:   const castI16again : i16 = @as(i16, constI32);
// CHECK-NEXT:   const castI32again : i32 = @as(i32, castI16);
// CHECK-NEXT:   const castU32 : u32 = @as(u32, constU16);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: task data_task(arg : f32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_data_task(data_task, @get_data_task_id(@get_color(0)));
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: task local_task() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_local_task(local_task, @get_local_task_id(1));
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: task control_task() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_control_task(control_task, @get_control_task_id(42));
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: task data_task_no_bind(arg0 : f32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: task local_task_no_bind() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: task control_task_no_bind() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: var uninit_array : [10]f32;
// CHECK-NEXT: var global_array : [10]f32 = @constants([10]f32, 4.2);
// CHECK-NEXT: const const_array : [10]i32 = @constants([10]i32, 10);

// CHECK-NEXT: var uninit_ptr : [*]f32 = &uninit_array;
// CHECK-NEXT: var global_ptr : [*]f32 = &global_array;
// CHECK-NEXT: const const_ptr : [*]const i32 = &const_array;
// CHECK-NEXT: var ptr_to_arr : *[10]f32 = &uninit_array;
// CHECK-NEXT: const ptr_to_val : *const i16 = &const27;
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
// CHECK-NEXT: //unknown op GetColorOp(%col = "csl.get_color"() <{"id" = 15 : i5}> : () -> !csl.color)
// CHECK-NEXT: //unknown op RpcOp("csl.rpc"(%col) : (!csl.color) -> ())

// CHECK-NEXT: var A : [24]f32 = @constants([24]f32, 0);
// CHECK-NEXT: var x : [6]f32 = @constants([6]f32, 0);
// CHECK-NEXT: var b : [4]f32 = @constants([4]f32, 0);
// CHECK-NEXT: var y : [4]f32 = @constants([4]f32, 0);
// CHECK-NEXT: const thing : imported_module = @import_module("<thing>");
// CHECK-NEXT:
// CHECK-NEXT: fn initialize() void {
// CHECK-NEXT:   const lb : i16 = 0;
// CHECK-NEXT:   const ub : i16 = 24;
// CHECK-NEXT:   const step : i16 = 1;
// CHECK-NEXT:   thing.some_func(lb, ub);
// CHECK-NEXT:   const res : i32 = thing.some_func(lb, ub);
// CHECK-NEXT:   const v1 : comptime_struct = thing.some_field;
// CHECK-NEXT:   const v2 : f32 = 3.14;
// CHECK-NEXT:   const v0 : f16 = 2.718;
// CHECK-NEXT:   const u32cst : u32 = 44;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub, step)) |idx| {
// CHECK-NEXT:     const idx_f32 : f32 = @as(f32, idx);
// CHECK-NEXT:     const idx_index : i32 = @as(i32, idx);
// CHECK-NEXT:     A[idx_index] = idx_f32;
// CHECK-NEXT:   }
// CHECK-NEXT:   const ub3 : i16 = 6;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub3, step)) |j| {
// CHECK-NEXT:     const val : f32 = 1.0;
// CHECK-NEXT:     const j_idx : i32 = @as(i32, j);
// CHECK-NEXT:     x[j_idx] = val;
// CHECK-NEXT:   }
// CHECK-NEXT:   const ub4 : i16 = 6;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub4, step)) |i| {
// CHECK-NEXT:     const c2 : f32 = 2.0;
// CHECK-NEXT:     const c0 : f32 = 0.0;
// CHECK-NEXT:     const i_idx : i32 = @as(i32, i);
// CHECK-NEXT:     b[i_idx] = c2;
// CHECK-NEXT:     y[i_idx] = c0;
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: fn gemv() void {
// CHECK-NEXT:   const lb : i32 = 0;
// CHECK-NEXT:   const step : i32 = 1;
// CHECK-NEXT:   const ub : i32 = 6;
// CHECK-NEXT:   const ub1 : i32 = 4;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i32, lb, ub1, step)) |i| {
// CHECK-NEXT:     const tmp : f32 = 0.0;
// CHECK-NEXT:     var tmp2 : f32 = tmp;
// CHECK-NEXT:
// CHECK-NEXT:     for(@range(i32, lb, ub, step)) |j| {
// CHECK-NEXT:       const ix6 : i32 = i * ub;
// CHECK-NEXT:       const ix6pj : i32 = ix6 +  j;
// CHECK-NEXT:       const Axx : f32 = (A[ix6pj]) * (x[j]);
// CHECK-NEXT:       tmp2 = tmp2 + Axx;
// CHECK-NEXT:     }
// CHECK-NEXT:     const tmp_plus_bi : f32 =  tmp2 + (b[i]);
// CHECK-NEXT:     y[i] = tmp_plus_bi;
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: // >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<< //
// CHECK-NEXT: //unknown op ParamOp(%p1 = "csl.param"() <{"param_name" = "param_1"}> : () -> i32)
// CHECK-NEXT: //unknown op ParamOp(%p2 = "csl.param"() <{"param_name" = "param_2", "init_value" = 1.300000e+00 : f16}> : () -> f16)
// CHECK-NEXT: layout {
// CHECK-NEXT:   const x_dim : i32 = 4;
// CHECK-NEXT:   const y_dim : i32 = 6;
// CHECK-NEXT:   //unknown op SetRectangleOp("csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ())
// CHECK-NEXT:   const x_coord0 : i32 = 0;
// CHECK-NEXT:   const y_coord : i32 = 0;
// CHECK-NEXT:   //unknown op SetTileCodeOp("csl.set_tile_code"(%x_coord0, %y_coord) <{"file" = "file.csl"}> : (i32, i32) -> ())
// CHECK-NEXT:   //unknown op ConstStructOp(%params = "csl.const_struct"() <{"items" = {"hello" = 123 : i32}}> : () -> !csl.comptime_struct)
// CHECK-NEXT:   const x_coord1 : i32 = 1;
// CHECK-NEXT:   //unknown op SetTileCodeOp("csl.set_tile_code"(%x_coord1, %y_coord, %params) <{"file" = "file.csl"}> : (i32, i32, !csl.comptime_struct) -> ())
// CHECK-NEXT:   @export_name("ptr_name", [*]f32, true);
// CHECK-NEXT:   @export_name("another_ptr", [*]const i32, false);
// CHECK-NEXT:   @export_name("no_args_no_return", fn() void, );
// CHECK-NEXT:   @export_name("args_no_return", fn(i32, i32) void, );
// CHECK-NEXT: }

// CHECK-EMPTY:
