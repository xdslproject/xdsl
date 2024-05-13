// RUN: xdsl-opt -t csl %s | filecheck  %s --match-full-lines

// Based on https://github.com/Cerebras/csl-examples/tree/master/tutorials/gemv-01-complete-program
module {

csl.module {kind = #csl<module_kind layout>} {
  %LAUNCH = "csl.get_color"() <{id = 8 : i5}> : () -> !csl.color
  %memcpy_init_params = "csl.const_struct"(%LAUNCH) <{
    items = { width = 1 : i32, height = 1 : i32},
    ssa_fields = ["LAUNCH"]
  }> : (!csl.color) -> !csl.comptime_struct
  %memcpy = "csl.import_module"(%memcpy_init_params) <{module = "<memcpy/get_params>"}> : (!csl.comptime_struct) -> !csl.comptime_struct
  csl.layout {
    %x_dim = arith.constant 1 : i32
    %y_dim = arith.constant 1 : i32
    "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()
    %x_coord = arith.constant 0 : i32
    %y_coord = arith.constant 0 : i32


    %memcpy_params = "csl.member_call"(%memcpy, %x_coord) <{field = "get_params"}> : (!csl.comptime_struct, i32) -> !csl.comptime_struct

    %tile_code_params = "csl.const_struct"(%memcpy_params) <{ssa_fields = ["memcpy_params"]}> : (!csl.comptime_struct) -> !csl.comptime_struct

    "csl.set_tile_code"(%x_coord, %y_coord, %tile_code_params) <{file = "pe_program.csl"}> : (i32, i32, !csl.comptime_struct) -> ()
  }
}

csl.module {kind = #csl<module_kind program>} {
  %memcpy_params = "csl.param"() <{param_name = "memcpy_params"}> : () -> !csl.comptime_struct

  %sys_mod = "csl.import_module"(%memcpy_params) <{module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.comptime_struct

  %M = arith.constant 4 : i16
  %N = arith.constant 6 : i16

"memref.global"() {"sym_name" = "A", "type" = memref<24xf32>, "sym_visibility" = "public", "initial_value"} : () -> ()
"memref.global"() {"sym_name" = "x", "type" = memref<6xf32>, "sym_visibility" = "public", "initial_value"} : () -> ()
"memref.global"() {"sym_name" = "b", "type" = memref<4xf32>, "sym_visibility" = "public", "initial_value"} : () -> ()
"memref.global"() {"sym_name" = "y", "type" = memref<4xf32>, "sym_visibility" = "public", "initial_value" = dense<0.0> : tensor<1xf32>} : () -> ()

  %A = memref.get_global @A : memref<24xf32>
  %x = memref.get_global @x : memref<6xf32>
  %b = memref.get_global @b : memref<4xf32>
  %y = memref.get_global @y : memref<4xf32>

  %y_ptr = "csl.addressof"(%y) : (memref<4xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const mut>>

  "csl.export"(%y_ptr) <{
    var_name = "y",
    type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const mut>>
  }> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const mut>>) -> ()


  csl.func @initialize() {
    %lb    = arith.constant 0 : index
    %step  = arith.constant 1 : index

    %ub_24 =arith.constant 24 : index
    scf.for %idx = %lb to %ub_24 step %step {
      %idx_i32 = "arith.index_cast"(%idx) : (index) -> i32
      %idx_f32 = "arith.sitofp"(%idx_i32) : (i32) -> f32
      memref.store %idx_f32, %A[%idx] : memref<24xf32>
    }

    %ub_6 =arith.constant 6 : index
    scf.for %idx = %lb to %ub_6 step %step {
      %val = arith.constant 1.0 : f32
      memref.store %val, %x[%idx] : memref<6xf32>
    }

    %ub_4 =arith.constant 4 : index
    scf.for %idx = %lb to %ub_4 step %step {
      %val = arith.constant 2.0 : f32
      memref.store %val, %b[%idx] : memref<4xf32>
    }


    csl.return
  }

  csl.func @gemv() {
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

  csl.func @init_and_compute() {
    "csl.call"() <{callee = @initialize}> : () -> ()
    "csl.call"() <{callee = @gemv}> : () -> ()

    "csl.member_call"(%sys_mod) <{field = "unblock_cmd_stream"}> : (!csl.comptime_struct) -> ()
    csl.return
  }

  "csl.export"() <{
    sym_name = @init_and_compute,
    type = () -> ()
  }> : () -> ()


  %LAUNCH =  "csl.member_access"(%sys_mod) <{field = "LAUNCH"}> : (!csl.comptime_struct) -> !csl.color
  "csl.rpc"(%LAUNCH) : (!csl.color) -> ()
}
}

// CHECK-NEXT: param memcpy_params : comptime_struct;
// CHECK-NEXT: const sys_mod : imported_module = @import_module("<memcpy/memcpy>", memcpy_params);
// CHECK-NEXT: const M : i16 = 4;
// CHECK-NEXT: const N : i16 = 6;
// CHECK-NEXT: var A : [24]f32;
// CHECK-NEXT: var x : [6]f32;
// CHECK-NEXT: var b : [4]f32;
// CHECK-NEXT: var y : [4]f32 = @constants([4]f32, 0.0);
// CHECK-NEXT: var y_ptr : [*]f32 = &y;
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(y_ptr, "y");
// CHECK-NEXT: }
// CHECK-NEXT: fn initialize() void {
// CHECK-NEXT:   const lb : i32 = 0;
// CHECK-NEXT:   const step : i32 = 1;
// CHECK-NEXT:   const ub_24 : i32 = 24;
// CHECK-NEXT:   for(@range(i32, lb, ub_24, step)) |idx| {
// CHECK-NEXT:     const idx_i32 : i32 = @as(i32, idx);
// CHECK-NEXT:     const idx_f32 : f32 = @as(f32, idx_i32);
// CHECK-NEXT:     A[idx] = idx_f32;
// CHECK-NEXT:   }
// CHECK-NEXT:   const ub_6 : i32 = 6;
// CHECK-NEXT:   for(@range(i32, lb, ub_6, step)) |idx0| {
// CHECK-NEXT:     const val : f32 = 1.0;
// CHECK-NEXT:     x[idx0] = val;
// CHECK-NEXT:   }
// CHECK-NEXT:   const ub_4 : i32 = 4;
// CHECK-NEXT:   for(@range(i32, lb, ub_4, step)) |idx1| {
// CHECK-NEXT:     const val : f32 = 2.0;
// CHECK-NEXT:     b[idx1] =  val;
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: fn gemv() void {
// CHECK-NEXT:   const lb : i32 = 0;
// CHECK-NEXT:   const step : i32 = 1;
// CHECK-NEXT:   const ub_6 : i32 = 6;
// CHECK-NEXT:   const ub_4 : i32 = 4;
// CHECK-NEXT:   for(@range(i32, lb, ub_4, step)) |i| {
// CHECK-NEXT:     const tmp_0 : f32 = 0.0;
// CHECK-NEXT:     var tmp : f32 = tmp_0;
// CHECK-NEXT:     for(@range(i32, lb, ub_6, step)) |j| {
// CHECK-NEXT:       const ix6 : i32 = i * ub_6;
// CHECK-NEXT:       const ix6pj : i32 = ix6 + j;
// CHECK-NEXT:       const Axx : f32 = (A[ix6pj]) * (x[j]);
// CHECK-NEXT:       tmp = tmp + Axx;
// CHECK-NEXT:     }
// CHECK-NEXT:     const tmp_plus_bi : f32 = tmp + (b[i]);
// CHECK-NEXT:     y[i] = tmp_plus_bi;
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: fn init_and_compute() void {
// CHECK-NEXT:   initialize();
// CHECK-NEXT:   gemv();
// CHECK-NEXT:   sys_mod.unblock_cmd_stream();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @export_symbol(init_and_compute, "init_and_compute");
// CHECK-NEXT: }
// CHECK-NEXT: const LAUNCH : color = sys_mod.LAUNCH;
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @rpc(@get_data_task_id(LAUNCH));
// CHECK-NEXT: }
// CHECK-NEXT: // >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<< //
// CHECK-NEXT: const LAUNCH0 : color = @get_color(8);
// CHECK-NEXT: const memcpy_init_params : comptime_struct = .{
// CHECK-NEXT:   .width = 1,
// CHECK-NEXT:   .height = 1,
// CHECK-NEXT:   .LAUNCH = LAUNCH0,
// CHECK-NEXT: };
// CHECK-NEXT: const memcpy : imported_module = @import_module("<memcpy/get_params>", memcpy_init_params);
// CHECK-NEXT: layout {
// CHECK-NEXT:   const x_dim : i32 = 1;
// CHECK-NEXT:   const y_dim : i32 = 1;
// CHECK-NEXT:   @set_rectangle(x_dim, y_dim);
// CHECK-NEXT:   const x_coord : i32 = 0;
// CHECK-NEXT:   const y_coord : i32 = 0;
// CHECK-NEXT:   const memcpy_params1 : comptime_struct = memcpy.get_params(x_coord);
// 
// CHECK-NEXT:   const tile_code_params : comptime_struct = .{
// CHECK-NEXT:     .memcpy_params = memcpy_params1,
// CHECK-NEXT:   };
// CHECK-NEXT:   @set_tile_code(x_coord, y_coord, "pe_program.csl", tile_code_params);
// CHECK-NEXT:   @export_name("y", [*]f32, true);
// CHECK-NEXT:   @export_name("init_and_compute", fn() void, );
// CHECK-NEXT:   }
// CHECK-EMPTY:
