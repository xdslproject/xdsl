// RUN: xdsl-opt -t csl %s | filecheck %s

"csl.module"() <{kind=#csl<module_kind program>}> ({

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


// CHECK-NEXT: var A : [24]f32 = @constants([24]f32, 0);
// CHECK-NEXT: var x : [6]f32 = @constants([6]f32, 0);
// CHECK-NEXT: var b : [4]f32 = @constants([4]f32, 0);
// CHECK-NEXT: var y : [4]f32 = @constants([4]f32, 0);
// CHECK-NEXT: const thing : imported_module = @import_module("<thing>");
// CHECK-NEXT:
// CHECK-NEXT: fn initialize() {
// CHECK-NEXT:   const lb : i16 = 0;
// CHECK-NEXT:   const ub : i16 = 24;
// CHECK-NEXT:   const step : i16 = 1;
// CHECK-NEXT:   thing.some_func(lb, ub);
// CHECK-NEXT:   const res : i32 = thing.some_func(lb, ub);
// CHECK-NEXT:   const v0 : comptime_struct = thing.some_field;
// CHECK-NEXT:   const v1 : f32 = 3.14;
// CHECK-NEXT:   const v02 : f16 = 2.718;
// CHECK-NEXT:   const u32cst : u32 = 44;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub, step)) |idx| {
// CHECK-NEXT:     //unknown op SIToFPOp(%idx_f32 = arith.sitofp %idx : i16 to f32
// CHECK-NEXT:     //unknown op IndexCastOp(%idx_index = "arith.index_cast"(%idx) : (i16) -> index)
// CHECK-NEXT:     //unknown op Store(memref.store %idx_f32, %A[%idx_index] : memref<24xf32>)
// CHECK-NEXT:   }
// CHECK-NEXT:   const ub3 : i16 = 6;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub3, step)) |j| {
// CHECK-NEXT:     const val : f32 = 1.0;
// CHECK-NEXT:     //unknown op IndexCastOp(%j_idx = "arith.index_cast"(%j) : (i16) -> index)
// CHECK-NEXT:     //unknown op Store(memref.store %val, %x[%j_idx] : memref<6xf32>)
// CHECK-NEXT:   }
// CHECK-NEXT:   const ub4 : i16 = 6;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub4, step)) |i| {
// CHECK-NEXT:     const c2 : f32 = 2.0;
// CHECK-NEXT:     const c0 : f32 = 0.0;
// CHECK-NEXT:     //unknown op IndexCastOp(%i_idx = "arith.index_cast"(%i) : (i16) -> index)
// CHECK-NEXT:     //unknown op Store(memref.store %c2, %b[%i_idx] : memref<4xf32>)
// CHECK-NEXT:     //unknown op Store(memref.store %c0, %y[%i_idx] : memref<4xf32>)
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT:
// CHECK-NEXT: fn gemv() {
// CHECK-NEXT:   const lb : i32 = 0;
// CHECK-NEXT:   const step : i32 = 1;
// CHECK-NEXT:   const ub : i32 = 6;
// CHECK-NEXT:   const ub0 : i32 = 4;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i32, lb, ub0, step)) |i| {
// CHECK-NEXT:     const tmp : f32 = 0.0;
// CHECK-NEXT:     var tmp1 : f32 = tmp;
// CHECK-NEXT:
// CHECK-NEXT:     for(@range(i32, lb, ub, step)) |j| {
// CHECK-NEXT:       //unknown op Muli(%ix6 = arith.muli %i, %ub : index)
// CHECK-NEXT:       //unknown op Addi(%ix6pj = arith.addi %ix6, %j : index)
// CHECK-NEXT:       //unknown op Load(%A_loaded = memref.load %A[%ix6pj] : memref<24xf32>)
// CHECK-NEXT:       //unknown op Load(%x_loaded = memref.load %x[%j] : memref<6xf32>)
// CHECK-NEXT:       //unknown op Mulf(%Axx = arith.mulf %A_loaded, %x_loaded : f32)
// CHECK-NEXT:       //unknown op Addf(%tmp_next = arith.addf %tmp_iter, %Axx : f32)
// CHECK-NEXT:     }
// CHECK-NEXT:     //unknown op Load(%bi = memref.load %b[%i] : memref<4xf32>)
// CHECK-NEXT:     //unknown op Addf(%tmp_plus_bi = arith.addf %tmp, %bi : f32)
// CHECK-NEXT:     //unknown op Store(memref.store %tmp_plus_bi, %y[%i] : memref<4xf32>)
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-EMPTY:
