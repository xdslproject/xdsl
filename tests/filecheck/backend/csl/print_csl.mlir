// RUN: xdsl-opt -t csl %s | filecheck %s

"memref.global"() {"sym_name" = "A", "type" = memref<24xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "x", "type" = memref<6xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "b", "type" = memref<4xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "y", "type" = memref<4xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()

%thing = "csl.import_module"() <{module = "<thing>"}> : () -> !csl.comptime_struct

csl.task @local_task() attributes {kind = #csl<task_kind local>, id = 0 : i5} {
  csl.return
}

csl.task @data_task(%a: i32) attributes {kind = #csl<task_kind data>, id = 2 : i5} {
  csl.return
}

// csl.task @control_task() attributes {kind = #csl.task_kind<control>, id = 42 : i6} {
//   csl.return
// }

csl.func @returning_func() -> i32 {
  %c = arith.constant 10 : i32
  csl.call @func_with_arguments(%c, %thing) : (i32, !csl.comptime_struct) -> ()
  csl.return %c : i32
}

csl.func @func_with_arguments(%a1: i32, %a2: !csl.comptime_struct) {
  %res = csl.call @returning_func() : () -> i32
  csl.return
}

csl.func @initialize() {
  %lb = arith.constant   0 : i16
  %ub = arith.constant  24 : i16
  %step = arith.constant 1 : i16

  // call without result
  "csl.member_call"(%thing, %lb, %ub) <{field = "some_func", operandSegmentSizes = array<i32: 1, 2>}> : (!csl.comptime_struct, i16, i16) -> ()

  // call with result
  %res = "csl.member_call"(%thing, %lb, %ub) <{field = "some_func", operandSegmentSizes = array<i32: 1, 2>}> : (!csl.comptime_struct, i16, i16) -> (i32)

  // member access
  %11 = "csl.member_access"(%thing) <{field = "some_field"}> : (!csl.comptime_struct) -> !csl.comptime_struct

  %0 = arith.constant 3.14 : f32
  %v0 = arith.constant 2.718 : f16

  %u32cst = arith.constant 44 : ui32

  %A = memref.get_global @A : memref<24xf32>
  %x = memref.get_global @x : memref<6xf32>
  %b = memref.get_global @b : memref<4xf32>
  %y = memref.get_global @y : memref<4xf32>

  scf.for %idx = %lb to %ub step %step {
    %idx_f32 = "arith.sitofp"(%idx) : (i16) -> f32
    %idx_index = "arith.index_cast"(%idx) : (i16) -> index
    memref.store %idx_f32, %A[%idx_index] : memref<24xf32>
  }

  %ub_6 = arith.constant 6 : i16

  scf.for %j = %lb to %ub step %step {
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


// CHECK-NEXT: //unknown op Global("memref.global"() <{"sym_name" = "A", "sym_visibility" = "public", "type" = memref<24xf32>, "initial_value" = dense<0> : tensor<1xindex>}> : () -> ())
// CHECK-NEXT: //unknown op Global("memref.global"() <{"sym_name" = "x", "sym_visibility" = "public", "type" = memref<6xf32>, "initial_value" = dense<0> : tensor<1xindex>}> : () -> ())
// CHECK-NEXT: //unknown op Global("memref.global"() <{"sym_name" = "b", "sym_visibility" = "public", "type" = memref<4xf32>, "initial_value" = dense<0> : tensor<1xindex>}> : () -> ())
// CHECK-NEXT: //unknown op Global("memref.global"() <{"sym_name" = "y", "sym_visibility" = "public", "type" = memref<4xf32>, "initial_value" = dense<0> : tensor<1xindex>}> : () -> ())
// CHECK-NEXT: const thing : comptime_struct = @import_module("<thing>");
// CHECK-NEXT: task local_task() void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_local_task(local_task, @get_local_task_id(0));
// CHECK-NEXT: }
// CHECK-NEXT: task data_task(a : i32) void {
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: comptime {
// CHECK-NEXT:   @bind_data_task(data_task, @get_data_task_id(@get_color(2)));
// CHECK-NEXT: }
// CHECK-NEXT: fn returning_func() i32 {
// CHECK-NEXT:   const c : i32 = 10;
// CHECK-NEXT:   func_with_arguments(c, thing);
// CHECK-NEXT:   return c;
// CHECK-NEXT: }
// CHECK-NEXT: fn func_with_arguments(a1 : i32, a2 : comptime_struct) void {
// CHECK-NEXT:   const res : i32 = returning_func();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT: fn initialize() void {
// CHECK-NEXT:   const lb : i16 = 0;
// CHECK-NEXT:   const ub : i16 = 24;
// CHECK-NEXT:   const step : i16 = 1;
// CHECK-NEXT:   thing.some_func(lb, ub);
// CHECK-NEXT:   const res : i32 = thing.some_func(lb, ub);
// CHECK-NEXT:   const v0 : comptime_struct = thing.some_field;
// CHECK-NEXT:   const v1 : f32 = 3.14;
// CHECK-NEXT:   const v02 : f16 = 2.718;
// CHECK-NEXT:   const u32cst : u32 = 44;
// CHECK-NEXT:   //unknown op GetGlobal(%A = memref.get_global @A : memref<24xf32>)
// CHECK-NEXT:   //unknown op GetGlobal(%x = memref.get_global @x : memref<6xf32>)
// CHECK-NEXT:   //unknown op GetGlobal(%b = memref.get_global @b : memref<4xf32>)
// CHECK-NEXT:   //unknown op GetGlobal(%y = memref.get_global @y : memref<4xf32>)
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub, step)) |idx| {
// CHECK-NEXT:     //unknown op SIToFPOp(%idx_f32 = "arith.sitofp"(%idx) : (i16) -> f32)
// CHECK-NEXT:     //unknown op IndexCastOp(%idx_index = "arith.index_cast"(%idx) : (i16) -> index)
// CHECK-NEXT:     //unknown op Store(memref.store %idx_f32, %A[%idx_index] : memref<24xf32>)
// CHECK-NEXT:     //unknown op Yield(scf.yield)
// CHECK-NEXT:   }
// CHECK-NEXT:   const ub_6 : i16 = 6;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub, step)) |j| {
// CHECK-NEXT:     const val : f32 = 1.0;
// CHECK-NEXT:     //unknown op IndexCastOp(%j_idx = "arith.index_cast"(%j) : (i16) -> index)
// CHECK-NEXT:     //unknown op Store(memref.store %val, %x[%j_idx] : memref<6xf32>)
// CHECK-NEXT:     //unknown op Yield(scf.yield)
// CHECK-NEXT:   }
// CHECK-NEXT:   const ub_4 : i16 = 6;
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, lb, ub_4, step)) |i| {
// CHECK-NEXT:     const c2 : f32 = 2.0;
// CHECK-NEXT:     const c0 : f32 = 0.0;
// CHECK-NEXT:     //unknown op IndexCastOp(%i_idx = "arith.index_cast"(%i) : (i16) -> index)
// CHECK-NEXT:     //unknown op Store(memref.store %c2, %b[%i_idx] : memref<4xf32>)
// CHECK-NEXT:     //unknown op Store(memref.store %c0, %y[%i_idx] : memref<4xf32>)
// CHECK-NEXT:     //unknown op Yield(scf.yield)
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
